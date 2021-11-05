import a0
import asyncio
import json


def unpack(pkt):
    reply = json.loads(pkt.payload.decode())
    if reply["err"] != "success":
        raise
    return reply["result"]


async def main():
    client = a0.AioRpcClient("panda_planner")

    box_pose = unpack(await client.send("move_group.get_random_pose()"))
    unpack(await client.send(f"scene.add_box('mybox', {box_pose}, (0.1, 0.1, 0.1))"))

    pose = unpack(await client.send("move_group.get_random_pose()"))
    unpack(await client.send(f"move_group.set_pose_target({pose})"))
    success, traj, planning_time, error_code = unpack(await client.send(f"move_group.plan()"))

    print(f"success: {success}")
    print(f"planning_time: {planning_time}")
    print(f"error_code: {error_code}")
    print(f"traj: {traj}")


asyncio.run(main())
