from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Which converter to start: 'apf' (default) or 'cbf'
    pipeline_arg = DeclareLaunchArgument(
        'pipeline',
        default_value='apf',
        description="Which ps5 converter to start: 'apf' or 'cbf'",
    )

    # Turn haptics on and off

    haptics_arg = DeclareLaunchArgument(
        'haptics',
        default_value='true',
        description="Turn haptics on or off",
    )

    # Read data from ps5 (always runs — both pipelines need the raw joystick)
    joy_node1 = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'dev': '/dev/input/js0',
            'deadzone': 0.15,
            'autorepeat_rate': 50.0,
        }],
        remappings=[
            ('/joy', '/ps5/joy')
        ]
    )

    # APF converter — Joy -> /controller/cmd_vel.  Only when pipeline == 'apf'.
    ps5_control_node = Node(
        package='ps5',
        executable='ps5_control_node',
        name='ps5_control_node',
        condition=LaunchConfigurationEquals('pipeline', 'apf'),
    )

    # CBF converter — Joy -> /controller/cmd_vel_raw.  Only when pipeline == 'cbf'.
    ps5_control_cbf_node = Node(
        package='ps5',
        executable='ps5_control_cbf',
        name='ps5_control_cbf_node',
        condition=LaunchConfigurationEquals('pipeline', 'cbf'),
    )

    # Direct converter — only when pipeline == 'direct'
    ps5_control_direct_node = Node(
        package='ps5',
        executable='ps5_control_direct',
        name='ps5_control_direct_node',
        condition=LaunchConfigurationEquals('pipeline', 'direct'),
    )

    # Haptic feedback (always runs — independent of pipeline)
    ps5_haptic_node = Node(
        package='ps5',
        executable='ps5_haptic',
        name='ps5_haptic_node',
        output='screen',
        condition=LaunchConfigurationEquals('haptics', 'true'),
    )



    return LaunchDescription([
        pipeline_arg,
        joy_node1,
        ps5_control_node,
        ps5_control_cbf_node,
        ps5_haptic_node,
        ps5_control_direct_node,
    ])