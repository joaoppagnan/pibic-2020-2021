docker run -p 8888:8888 --gpus all -it -v $(pwd):/pibic2020-docker --rm pibic2020-docker python3.8 -m jupyterlab --ip='0.0.0.0' --port=8888 --no-browser --allow-root
