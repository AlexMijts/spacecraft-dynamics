from setuptools import setup

setup(
    name="spacecraftDynamics",
    packages=["attitude_coordinates", "attitude_estimation", "nl_control"],
    install_requires=["numpy"]
)
