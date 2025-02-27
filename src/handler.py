import runpod
import subprocess
import os
import base64
from runpod.serverless.utils import rp_upload

# Define paths
YU_E_DIR = "YuE/inference/"
OUTPUT_DIR = "/output"


def clean_output_dir():
    """ Removes all files in the output directory before each run. """
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(OUTPUT_DIR)


def encode_files(guid):
    """ Encodes all generated files in the output directory as Base64. """
    encoded_files = []
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                if os.environ.get("BUCKET_ENDPOINT_URL"):
                    log(f"Uploading mp3 to bucket...")
                    url = rp_upload.upload_file_to_bucket(f"{guid}.mp3", file_path)
                    log(f"Uploaded mp3 to {url}")
                    encode_files.append({"filename": file, "url": url})
                else:
                    with open(file_path, "rb") as f:
                        encoded_content = base64.b64encode(f.read()).decode("utf-8")
                        encoded_files.append({"filename": file, "content": encoded_content})
    return encoded_files


def run_command(command):
    """ Run a shell command and capture output. Wait for the process to finish before moving on. Log each line of output and stderror """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    error = ""
    for line in process.stdout:
        log(line)
    for line in process.stderr:
        log(line)
        error += line + "\n"
    process.wait()
    # return true if successful false if error
    return process.returncode == 0, error

def log(message):
    """ Log a message to the console. """
    print(f'[runpod-yuegp-worker] {message}')

def handler(job):
    """ Handler function that processes jobs. """
    job_input = job["input"]

    # Clean output directory before starting
    clean_output_dir()

    # Extract common parameters
    genre_txt = job_input.get("genre_txt", "a rock song")
    # write the genre_txt to a file
    with open(f"{YU_E_DIR}/genre.txt", "w") as f:
        f.write(genre_txt)
    lyrics_txt = job_input.get("lyrics_txt", "some lyrics")
    # write the lyrics_txt to a file
    with open(f"{YU_E_DIR}/lyrics.txt", "w") as f:
        f.write(lyrics_txt)
    run_n_segments = job_input.get("run_n_segments", 2)
    stage2_batch_size = job_input.get("stage2_batch_size", 4)
    max_new_tokens = job_input.get("max_new_tokens", 3000)
    cuda_idx = job_input.get("cuda_idx", 0)

    # Check for reference audio
    audio_prompt_path = job_input.get("audio_prompt_path", None)
    prompt_start_time = job_input.get("prompt_start_time", 0)
    prompt_end_time = job_input.get("prompt_end_time", 30)

    # Choose models based on input type
    if audio_prompt_path:
        # Use reference audio
        stage1_model = "m-a-p/YuE-s1-7B-anneal-en-icl"
    else:
        # Text generation
        stage1_model = "m-a-p/YuE-s1-7B-anneal-en-cot"

    stage2_model = "m-a-p/YuE-s2-1B-general"

    # Construct base command
    command = (
        f"cd {YU_E_DIR} && python -u ./infer.py "
        f"--stage1_model {stage1_model} "
        f"--stage2_model {stage2_model} "
        f"--genre_txt genre.txt "
        f"--lyrics_txt lyrics.txt "
        f"--run_n_segments {run_n_segments} "
        f"--stage2_batch_size {stage2_batch_size} "
        f"--output_dir {OUTPUT_DIR} "
        f"--cuda_idx {cuda_idx} "
        f"--max_new_tokens {max_new_tokens} "
    )

    # Add audio-specific parameters if provided
    if audio_prompt_path:
        log(f"Using audio prompt: {audio_prompt_path}")
        command += (
            f"--audio_prompt_path {audio_prompt_path} "
            f"--prompt_start_time {prompt_start_time} "
            f"--prompt_end_time {prompt_end_time} "
        )

    log(f"Running command: {command}")
    # Run command
    status, error = run_command(command)
    if not status:
        return {
            "status": "error",
            "error": "Failed to generate music",
            "message": error
        }
     # if there is a job_input['id'] then use that for the guid, otherwise generate one
    if 'id' in job_input:
        guid = job_input['id']
    else:
        import uuid
        guid = str(uuid.uuid4())

    log("Encoding generated files...")
    # Encode generated files
    encoded_files = encode_files(guid)

    # Return response
    return {
        "status": "completed",
        "generated_files": encoded_files,
    }


runpod.serverless.start({"handler": handler})
