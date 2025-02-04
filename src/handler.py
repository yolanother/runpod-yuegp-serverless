import runpod
import subprocess
import os
import base64

# Define paths
YU_E_DIR = "/YuE/inference/"
OUTPUT_DIR = "./output"


def clean_output_dir():
    """ Removes all files in the output directory before each run. """
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(OUTPUT_DIR)


def encode_files():
    """ Encodes all generated files in the output directory as Base64. """
    encoded_files = []
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    encoded_content = base64.b64encode(f.read()).decode("utf-8")
                    encoded_files.append({"filename": file, "content": encoded_content})
    return encoded_files


def run_command(command):
    """ Run a shell command and capture output. """
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    return process.stdout, process.stderr


def handler(job):
    """ Handler function that processes jobs. """
    job_input = job["input"]

    # Clean output directory before starting
    clean_output_dir()

    # Extract common parameters
    genre_txt = job_input.get("genre_txt", "prompt_examples/genre.txt")
    lyrics_txt = job_input.get("lyrics_txt", "prompt_examples/lyrics.txt")
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
        f"cd {YU_E_DIR} && python infer.py "
        f"--stage1_model {stage1_model} "
        f"--stage2_model {stage2_model} "
        f"--genre_txt {genre_txt} "
        f"--lyrics_txt {lyrics_txt} "
        f"--run_n_segments {run_n_segments} "
        f"--stage2_batch_size {stage2_batch_size} "
        f"--output_dir {OUTPUT_DIR} "
        f"--cuda_idx {cuda_idx} "
        f"--max_new_tokens {max_new_tokens} "
    )

    # Add audio-specific parameters if provided
    if audio_prompt_path:
        command += (
            f"--audio_prompt_path {audio_prompt_path} "
            f"--prompt_start_time {prompt_start_time} "
            f"--prompt_end_time {prompt_end_time} "
        )

    # Run command
    stdout, stderr = run_command(command)

    # Encode generated files
    encoded_files = encode_files()

    # Return response
    return {
        "status": "completed",
        "stdout": stdout,
        "stderr": stderr,
        "generated_files": encoded_files,
    }


runpod.serverless.start({"handler": handler})
