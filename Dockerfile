FROM public.ecr.aws/z9b3d4t5/jobs-container-keras-export-base:d9311cab7049527a58a3fed6791aa7066e3ead53

WORKDIR /scripts

# Install extra dependencies here
COPY requirements.txt ./
RUN /app/keras/.venv/bin/pip3 install --no-cache-dir -r requirements.txt

# Copy all files to the home directory
COPY . ./

# The train command (we run this from the keras venv, which has all dependencies)
ENTRYPOINT [ "./run-python-with-venv.sh", "keras", "train.py" ]
