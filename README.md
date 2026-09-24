# kairos_ws Docker workflow

ROS Noetic dev container for the `optimize_visualize` PS5/DualSense work.
The image bakes in all the recurring installs; the workspace lives on the host
and is mounted in, so you edit on the host and build inside the container.

---

## One-time setup

Both files (`Dockerfile`, `run_container.sh`) should sit in
`/home/developer/kairos_ws/` on the host.

```bash
cd /home/developer/kairos_ws

# 1. Build the image (installs pip, hidapi, pydualsense — done once)
docker build -t ros-noetic-ds .

# 2. If an old container with the same name exists, remove it
docker rm ros-noetic 2>/dev/null

# 3. Create + start the container for the first time
chmod +x run_container.sh
./run_container.sh
```

You're now inside the container. ROS is already sourced. `pip`, `hidapi`,
and `pydualsense` are already installed.

---

## Daily workflow

```bash
# --- on the HOST ---
docker start -i ros-noetic        # boot the existing container

# --- inside the CONTAINER ---
build_ws                          # rebuild + source (only when code changed)
rosrun cv_basics joy_repacker.py  # run your node
```

Open extra shells into the running container (e.g. one for roscore, one for a node):

```bash
docker exec -it ros-noetic bash
```

Shut down: type `exit` in the original shell, or `docker stop ros-noetic` from the host.

---

## What's automated

| Thing                                   | Where        | When it runs            |
|-----------------------------------------|--------------|-------------------------|
| pip, libhidapi, pydualsense             | Dockerfile   | once, at image build    |
| `source /opt/ros/noetic/setup.bash`     | /root/.bashrc| every shell             |
| source workspace overlay (if built)     | /root/.bashrc| every shell             |
| `build_ws` alias (catkin_make + source) | /root/.bashrc| on demand, when you type it |
| privileged + /dev passthrough (DualSense)| run_container.sh | at container create |
| DNS fix, X11, host networking, mount    | run_container.sh | at container create |

---

## Notes

- **`build_ws` is manual on purpose.** Auto-building on every shell would be slow
  and could collide across multiple `exec` shells. Run it only after you change code.
- **`docker rm` wipes the container** but NOT your work — `kairos_ws` lives on the
  host. After a recreate you don't reinstall anything (it's in the image); just
  run `build_ws` once to rebuild the workspace overlay.
- **Adding a new pip/apt dependency?** Add it to the `Dockerfile` and rerun
  `docker build -t ros-noetic-ds .` so it stays reproducible — don't install it
  ad-hoc inside the container, or you'll lose it on the next recreate.
- **Controller not opening?** Confirm `/dev/hidraw*` is visible *inside* the
  container (`ls -l /dev/hidraw*`). If missing, the controller was likely plugged
  in after launch — `docker stop ros-noetic && docker start -i ros-noetic`, or
  recreate.
