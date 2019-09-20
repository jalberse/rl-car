# Gotta go Fast

## Getting Started

Running the racecar environment requires a number of dependencies, including mujoco. More information is on the [README for gym under Installation](https://github.com/openai/gym). This process may take a few days, as they authorize accounts by hand. 

Once mujoco is set up, enter the shell with `pipenv shell`

If this step fails, you may lack some dependencies. Try the following commands and any from the openai/mujoco troubleshooting guides.

```
sudo apt install swig
```

We use OpenCV for image processing. Follow the tutorial for [installing OpenCV on Ubuntu](https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html)

## Playing as a human

In the shell, run `python car_racing.py` to play the race as a human.
