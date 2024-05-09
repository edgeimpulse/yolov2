import sys, os, shutil, signal, random, operator, functools, time, subprocess, math, contextlib, io, skimage, argparse
import logging, threading

from frontend import YOLO

dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Edge Impulse training scripts')
parser.add_argument('--info-file', type=str, required=False,
                    help='train_input.json file with info about classes and input shape',
                    default=os.path.join(dir_path, 'train_input.json'))
parser.add_argument('--data-directory', type=str, required=True,
                    help='Where to read the data from')
parser.add_argument('--out-directory', type=str, required=True,
                    help='Where to write the data')

parser.add_argument('--epochs', type=int, required=False,
                    help='Number of training cycles')
parser.add_argument('--learning-rate', type=float, required=False,
                    help='Learning rate')
parser.add_argument('--batch_size', type=int, required=False,
                    help='Training batch size')
parser.add_argument('--ensure-determinism', action='store_true',
                    help='Prevent non-determinism, e.g. do not shuffle batches')

args, unknown = parser.parse_known_args()

# Info about the training pipeline (inputs / shapes / modes etc.)
if not os.path.exists(args.info_file):
    print('Info file', args.info_file, 'does not exist')
    exit(1)

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.disable(logging.WARNING)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import numpy as np

# Suppress Numpy deprecation warnings
# TODO: Only suppress warnings in production, not during development
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# Filter out this erroneous warning (https://stackoverflow.com/a/70268806 for context)
warnings.filterwarnings('ignore', 'Custom mask layers require a config and must override get_config')

RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

# Since it also includes TensorFlow and numpy, this library should be imported after TensorFlow has been configured
sys.path.append('./resources/libraries')
import ei_tensorflow.training
import ei_tensorflow.conversion
import ei_tensorflow.profiling
import ei_tensorflow.inference
import ei_tensorflow.embeddings
import ei_tensorflow.brainchip.model
import ei_tensorflow.gpu
from ei_shared.parse_train_input import parse_train_input, parse_input_shape


import json, datetime, time, traceback
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error

input = parse_train_input(args.info_file)

# Small hack to detect on profile stage if Custom Learning Block produced Akida model
# On creating profile script, Studio is not aware that CLB can produce Akida model
if os.path.exists(os.path.join(args.out_directory, 'akida_model.fbz')):
    input.akidaModel = True

# For SSD object detection models we specify batch size via 'input'.
# We must allow it to be overridden via command line args:
if input.mode == 'object-detection' and 'batch_size' in args and args.batch_size is not None:
    input.objectDetectionBatchSize = args.batch_size or 16

BEST_MODEL_PATH = os.path.join(os.sep, 'tmp', 'best_model.tf' if input.akidaModel else 'best_model.hdf5')

# Information about the data and input:
# The shape of the model's input (which may be different from the shape of the data)
MODEL_INPUT_SHAPE = parse_input_shape(input.inputShapeString)
# The length of the model's input, used to determine the reshape inside the model
MODEL_INPUT_LENGTH = MODEL_INPUT_SHAPE[0]
MAX_TRAINING_TIME_S = input.maxTrainingTimeSeconds
MAX_GPU_TIME_S = input.remainingGpuComputeTimeSeconds

online_dsp_config = None

if (online_dsp_config != None):
    print('The online DSP experiment is enabled; training will be slower than normal.')

# load imports dependening on import
if (input.mode == 'object-detection' and input.objectDetectionLastLayer == 'mobilenet-ssd'):
    import ei_tensorflow.object_detection

def exit_gracefully(signum, frame):
    print("")
    print("Terminated by user", flush=True)
    time.sleep(0.2)
    sys.exit(1)


def train_model(train_dataset, validation_dataset, input_length, callbacks,
                X_train, X_test, Y_train, Y_test, train_sample_count, classes, classes_values,
                ensure_determinism=False):
    global ei_tensorflow

    disable_per_channel_quantization = False
    # We can optionally output a Brainchip Akida pre-trained model
    akida_model = None
    akida_edge_model = None

    if (input.mode == 'object-detection' and input.objectDetectionLastLayer == 'mobilenet-ssd'):
        ei_tensorflow.object_detection.set_limits(max_training_time_s=MAX_TRAINING_TIME_S,
            max_gpu_time_s=MAX_GPU_TIME_S,
            is_enterprise_project=input.isEnterpriseProject)

    
    import os
    import tensorflow as tf
    import numpy as np
    from akida_models import akidanet_imagenet
    from keras import Model
    from tensorflow.keras.optimizers.legacy import Adam
    from tensorflow.keras.layers import BatchNormalization, Conv2D, Softmax, ReLU
    from cnn2snn import check_model_compatibility
    from ei_tensorflow.constrained_object_detection import models, dataset, metrics, util
    
    WEIGHTS_PREFIX = os.environ.get('WEIGHTS_PREFIX', os.getcwd())
    
    def build_model(input_shape: tuple, alpha: float,
                    num_classes: int, weight_regularizer=None) -> tf.keras.Model:
        """ Construct a constrained object detection model.
    
        Args:
            input_shape: Passed to AkidaNet construction.
            alpha: AkidaNet alpha value.
            num_classes: Number of classes, i.e. final dimension size, in output.
    
        Returns:
            Uncompiled keras model.
    
        Model takes (B, H, W, C) input and
        returns (B, H//8, W//8, num_classes) logits.
        """
   
        yolo = YOLO(backend             = 'Tiny Yolo',
                    input_size          = 224, 
                    labels              = ['cat', 'car'], 
                    max_box_per_image   = 10,
                    anchors             = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
           
        #! Check if the model is sompatbile with Akida (fail quickly before training)
        compatible = check_model_compatibility(yolo, input_is_image=True)
        if not compatible:
            print("Model is not compatible with Akida!")
            sys.exit(1)
    
        return yolo
    
    def train(num_classes: int, learning_rate: float, num_epochs: int,
                alpha: float, object_weight: int,
                train_dataset: tf.data.Dataset,
                validation_dataset: tf.data.Dataset,
                best_model_path: str,
                input_shape: tuple,
                callbacks: 'list',
                quantize_function,
                qat_function,
                lr_finder: bool = False) -> tf.keras.Model:
        """ Construct and train a constrained object detection model.
    
        Args:
            num_classes: Number of classes in datasets. This does not include
                implied background class introduced by segmentation map dataset
                conversion.
            learning_rate: Learning rate for Adam.
            num_epochs: Number of epochs passed to model.fit
            alpha: Alpha used to construct AkidaNet. Pretrained weights will be
                used if there is a matching set.
            object_weight: The weighting to give the object in the loss function
                where background has an implied weight of 1.0.
            train_dataset: Training dataset of (x, (bbox, one_hot_y))
            validation_dataset: Validation dataset of (x, (bbox, one_hot_y))
            best_model_path: location to save best model path. note: weights
                will be restored from this path based on best val_f1 score.
            input_shape: The shape of the model's input
            lr_finder: TODO
        Returns:
            Trained keras model.
    
        Constructs a new constrained object detection model with num_classes+1
        outputs (denoting the classes with an implied background class of 0).
        Both training and validation datasets are adapted from
        (x, (bbox, one_hot_y)) to (x, segmentation_map). Model is trained with a
        custom weighted cross entropy function.
        """
    
    
        num_classes_with_background = num_classes + 1
    
        input_width_height = None
        width, height, input_num_channels = input_shape
        if width != height:
            raise Exception(f"Only square inputs are supported; not {input_shape}")
        input_width_height = width
    
        yolo = build_model(input_shape=input_shape,
                            alpha=alpha,
                            num_classes=num_classes_with_background,
                            weight_regularizer=tf.keras.regularizers.l2(4e-5))
    
        #! Derive output size from model
        model_output_shape = model.layers[-1].output.shape
        _batch, width, height, num_classes = model_output_shape
        if width != height:
            raise Exception(f"Only square outputs are supported; not {model_output_shape}")
        output_width_height = width
    
        yolo.train(train_imgs     = train_imgs,
                valid_imgs         = valid_imgs,
                train_times        = 8,
                valid_times        = 1,
                nb_epochs          = 1, 
                learning_rate      = learning_rate, 
                batch_size         = 16,
                warmup_epochs      = 3,
                object_scale       = 5.0,
                no_object_scale    = 1.0,
                coord_scale        = 1.0,
                class_scale        = 1.0,
                saved_weights_name = 'save_yolo_weights.h5',
                debug              = True)

    
        #! Check if model is compatible with Akida
        compatible = check_model_compatibility(yolo, input_is_image=True)
        if not compatible:
            print("Model is not compatible with Akida!")
            sys.exit(1)
    
        #! Quantize model to 4/4/8
        akida_model = quantize_function(keras_model=yolo)
    
        #! Perform quantization-aware training
        akida_model = qat_function(akida_model=akida_model,
                                   train_dataset=train_segmentation_dataset,
                                   validation_dataset=validation_segmentation_dataset,
                                   optimizer=opt,
                                   fine_tune_loss=weighted_xent,
                                   fine_tune_metrics=None,
                                   callbacks=callbacks + [centroid_callback, print_callback],
                                   stopping_metric='val_f1',
                                   fit_verbose=0)
    
        return yolo, akida_model
    
    
    EPOCHS = args.epochs or 20
    LEARNING_RATE = args.learning_rate or 0.001
    
    import tensorflow as tf
    
    
    def akida_quantize_model(
        keras_model,
        weight_quantization: int = 4,
        activ_quantization: int = 4,
        input_weight_quantization: int = 8,
    ):
        import cnn2snn
    
        print("Performing post-training quantization...")
        akida_model = cnn2snn.quantize(
            keras_model,
            weight_quantization=weight_quantization,
            activ_quantization=activ_quantization,
            input_weight_quantization=input_weight_quantization,
        )
        print("Performing post-training quantization OK")
        print("")
    
        return akida_model
    
    
    def akida_perform_qat(
        akida_model,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        optimizer: str,
        fine_tune_loss: str,
        fine_tune_metrics: "list[str]",
        callbacks,
        stopping_metric: str = "val_accuracy",
        fit_verbose: int = 2,
        qat_epochs: int = 60,
    ):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=stopping_metric,
            mode="max",
            verbose=1,
            min_delta=0,
            patience=10,
            restore_best_weights=True,
        )
        callbacks.append(early_stopping)
        
        optimizer = Adam(learning_rate=LEARNING_RATE/100)
    
        print("Running quantization-aware training...")
        akida_model.compile(
            optimizer=optimizer, loss=fine_tune_loss, metrics=fine_tune_metrics
        )
    
        akida_model.fit(
            train_dataset,
            epochs=qat_epochs,
            verbose=fit_verbose,
            validation_data=validation_dataset,
            callbacks=callbacks,
        )
    
        print("Running quantization-aware training Done")
        print("")
    
        return akida_model
    
    
    model, akida_model = train(num_classes=classes,
                               learning_rate=LEARNING_RATE,
                               num_epochs=EPOCHS,
                               alpha=0.5,
                               object_weight=100,
                               train_dataset=train_dataset,
                               validation_dataset=validation_dataset,
                               best_model_path=BEST_MODEL_PATH,
                               input_shape=MODEL_INPUT_SHAPE,
                               callbacks=callbacks,
                               quantize_function=akida_quantize_model,
                               qat_function=akida_perform_qat,
                               lr_finder=False)
    
    override_mode = 'segmentation'
    disable_per_channel_quantization = False
    return model, disable_per_channel_quantization, akida_model, akida_edge_model

# This callback ensures the frontend doesn't time out by sending a progress update every interval_s seconds.
# This is necessary for long running epochs (in big datasets/complex models)
class BatchLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, train_sample_count, epochs, interval_s = 10, ensure_determinism=False):
        # train_sample_count could be smaller than the batch size, so make sure total_batches is atleast
        # 1 to avoid a 'divide by zero' exception in the 'on_train_batch_end' callback.
        self.total_batches = max(1, int(train_sample_count / batch_size))
        self.last_log_time = time.time()
        self.epochs = epochs
        self.interval_s = interval_s
        print(f'Using batch size: {batch_size}', flush=True)

    # Within each epoch, print the time every 10 seconds
    def on_train_batch_end(self, batch, logs=None):
        current_time = time.time()
        if self.last_log_time + self.interval_s < current_time:
            print('Epoch {0}% done'.format(int(100 / self.total_batches * batch)), flush=True)
            self.last_log_time = current_time

    # Reset the time the start of every epoch
    def on_epoch_end(self, epoch, logs=None):
        self.last_log_time = time.time()

def main_function():
    """This function is used to avoid contaminating the global scope"""
    classes_values = input.classes
    classes = 1 if input.mode == 'regression' else len(classes_values)

    mode = input.mode
    object_detection_last_layer = input.objectDetectionLastLayer if input.mode == 'object-detection' else None

    train_dataset, validation_dataset, samples_dataset, X_train, X_test, Y_train, Y_test, has_samples, X_samples, Y_samples = ei_tensorflow.training.get_dataset_from_folder(
        input, args.data_directory, RANDOM_SEED, online_dsp_config, MODEL_INPUT_SHAPE, args.ensure_determinism
    )

    callbacks = ei_tensorflow.training.get_callbacks(dir_path, mode, BEST_MODEL_PATH,
        object_detection_last_layer=object_detection_last_layer,
        is_enterprise_project=input.isEnterpriseProject,
        max_training_time_s=MAX_TRAINING_TIME_S,
        max_gpu_time_s=MAX_GPU_TIME_S,
        enable_tensorboard=input.tensorboardLogging)

    model = None

    print('')
    print('Training model...')
    ei_tensorflow.gpu.print_gpu_info()
    print('Training on {0} inputs, validating on {1} inputs'.format(len(X_train), len(X_test)))
    # USER SPECIFIC STUFF
    model, disable_per_channel_quantization, akida_model, akida_edge_model = train_model(train_dataset, validation_dataset,
        MODEL_INPUT_LENGTH, callbacks, X_train, X_test, Y_train, Y_test, len(X_train), classes, classes_values, args.ensure_determinism)
    # END OF USER SPECIFIC STUFF

    # REST OF THE APP
    print('Finished training', flush=True)
    print('', flush=True)

    # Make sure these variables are here, even when quantization fails
    tflite_quant_model = None

    if mode == 'object-detection':
        if input.objectDetectionLastLayer != 'fomo':
            tflite_model, tflite_quant_model = ei_tensorflow.object_detection.convert_to_tf_lite(
                args.out_directory,
                saved_model_dir='saved_model',
                validation_dataset=validation_dataset,
                model_filenames_float='model.tflite',
                model_filenames_quantised_int8='model_quantized_int8_io.tflite')
        else:
            from ei_tensorflow.constrained_object_detection.conversion import convert_to_tf_lite
            tflite_model, tflite_quant_model = convert_to_tf_lite(
                args.out_directory, model,
                saved_model_dir='saved_model',
                h5_model_path='model.h5',
                validation_dataset=validation_dataset,
                model_filenames_float='model.tflite',
                model_filenames_quantised_int8='model_quantized_int8_io.tflite',
                disable_per_channel=disable_per_channel_quantization)
            if input.akidaModel:
                if not akida_model:
                    print('Akida training code must assign a quantized model to a variable named "akida_model"', flush=True)
                    exit(1)
                ei_tensorflow.brainchip.model.convert_akida_model(args.out_directory, akida_model,
                                                                'akida_model.fbz',
                                                                MODEL_INPUT_SHAPE)
    else:
        model, tflite_model, tflite_quant_model = ei_tensorflow.conversion.convert_to_tf_lite(
            model, BEST_MODEL_PATH, args.out_directory,
            saved_model_dir='saved_model',
            h5_model_path='model.h5',
            validation_dataset=validation_dataset,
            model_input_shape=MODEL_INPUT_SHAPE,
            model_filenames_float='model.tflite',
            model_filenames_quantised_int8='model_quantized_int8_io.tflite',
            disable_per_channel=disable_per_channel_quantization,
            syntiant_target=input.syntiantTarget,
            akida_model=input.akidaModel)

        if input.akidaModel:
            if not akida_model:
                print('Akida training code must assign a quantized model to a variable named "akida_model"', flush=True)
                exit(1)

            ei_tensorflow.brainchip.model.convert_akida_model(args.out_directory, akida_model,
                                                              'akida_model.fbz',
                                                              MODEL_INPUT_SHAPE)
            if input.akidaEdgeModel:
                ei_tensorflow.brainchip.model.convert_akida_model(args.out_directory, akida_edge_model,
                                                                'akida_edge_learning_model.fbz',
                                                                MODEL_INPUT_SHAPE)
            else:
                import os
                model_full_path = os.path.join(args.out_directory, 'akida_edge_learning_model.fbz')
                if os.path.isfile(model_full_path):
                    os.remove(model_full_path)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    main_function()