# Use the official ROS 2 Humble desktop image
FROM osrf/ros:humble-desktop

# Install standard dependencies for ROS 2, joystick, and audio
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-joy \
    ros-humble-tf-transformations \
    python3-transforms3d \
    joystick \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip3 install quadprog sounddevice

# Set the working directory inside the container
WORKDIR /root/safety_main_cbf

# Copy everything from your local safety_main_cbf folder into the container
COPY . /root/safety_main_cbf/

# Build the ps5_ws workspace
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    cd ps5_ws && \
    colcon build"

# Automatically source ROS 2 and your workspace when the container starts
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /root/safety_main_cbf/ps5_ws/install/setup.bash" >> ~/.bashrc

# Set the default command to open a bash shell
CMD ["/bin/bash"]
