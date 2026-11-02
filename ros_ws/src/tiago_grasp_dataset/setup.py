from glob import glob
from setuptools import find_packages, setup


package_name = "tiago_grasp_dataset"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TIAGo Grasp",
    maintainer_email="local@example.com",
    description="Decoupled tabletop simulation and optional grasp policies for TIAGo.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "scene_manager = tiago_grasp_dataset.scene:main",
            "run_heuristic = tiago_grasp_dataset.generator:main",
            "replay_grasp = tiago_grasp_dataset.replay:main",
            "capture_calibration = tiago_grasp_dataset.calibration_capture:main",
        ],
    },
)
