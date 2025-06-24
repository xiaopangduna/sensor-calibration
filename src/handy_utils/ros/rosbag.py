#!C:/A_Programme/Miniconda/envs/py310 python
# -*- coding: utf-8 -*-
"""
@File    :   rosbag2video.py
@Time    :   2024/04/20 17:18:10
@Author  :   xiaopangdun
@Email  :   18675381281@163.com
@Version :   1.0
@Desc    :   None
"""

# import sys

# import os

# ros_cv2_paths = [
#     "/opt/ros/humble/local/lib/python3.10/dist-packages",
#     "/opt/ros/humble/lib/python3.10/site-packages",
#     "/usr/lib/python3/dist-packages",
# ]
# for path in ros_cv2_paths:
#     if path in sys.path:
#         sys.path.remove(path)
# print(sys.path)
# sys.path.append("/usr/lib/python3/dist-packages")
from pathlib import Path

import cv2
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Typestore


class RosBagProcessor(object):

    @staticmethod
    def rosbag2video(
        path_bag: Path,
        path_video: Path,
        topic: str,
        img_size: tuple,
        fourcc=cv2.VideoWriter_fourcc(*"XVID"),
        frame_rate: int = 30,
        typestore: Typestore = get_typestore(Stores.ROS2_HUMBLE),
    ):
        """Extract images from rosbag and merge them into videos.

        Args:
            path_bag (Path): The path to save ros bag.
            path_video (Path): The path to save video.
            topic (str): The name of ros topic.
            img_size (tuple): The tuple of (W,H),where W is the weight of video,where H is the height of video.
            fourcc (_type_, optional): The coding mode if video. Defaults to cv2.VideoWriter_fourcc(*"XVID").
            typestore (Typestore, optional): The message types of ros. Defaults to get_typestore(Stores.ROS2_HUMBLE).
            frame_rate (int, optional): The frame rate of video. Defaults to 30.
        Example:
            paths = [Path(r"D:/A_Project/database/bus_data/bag/240316/14_20"),]
            dir_video = Path(r"D:/A_Project/database/bus_data/temp")
            img_size = (1920, 1536)  # WH
            topic = "/platform/camera_0/image_compressed"
            frame_rate = 30
            typestore = get_typestore(Stores.ROS2_HUMBLE)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            processor = RosBagProcessor()
            for path in paths:
                path_video = dir_video.joinpath(path.name + ".avi")
                processor.rosbag2video(
                    path, path_video, topic, img_size, fourcc, frame_rate, typestore
                )
        """
        print("-" * 70)
        print("Start extract video from rosbag")
        out = cv2.VideoWriter(
            str(path_video), fourcc, frame_rate, img_size, True
        )
        # Create reader instance and open for reading.
        with AnyReader([path_bag], default_typestore=typestore) as reader:
            connections = [x for x in reader.connections if x.topic == topic]
            count = 0
            for connection, timestamp, rawdata in reader.messages(
                connections=connections
            ):

                count += 1
                msg = reader.deserialize(rawdata, connection.msgtype)
                img = message_to_cvimage(msg)
                if img.shape[0] != img_size[1] or img.shape[1] != img_size[0]:
                    img = cv2.resize(img, img_size)

                out.write(img)
                # print(
                #     "Extract {}/{} frame from {} to {}".format(
                #         count, connection.msgcount, path_bag, path_video
                #     )
                # )

            out.release()
            # img.shpae HWC
            if img.shape[0] != img_size[1] or img.shape[1] != img_size[0]:
                print(
                    "Warn : Image sizes are inconsistent: image size {}*{} != video size {}*{}".format(
                        img.shape[1],
                        img.shape[0],
                        img_size[0],
                        img_size[1],
                    )
                )
        print(
            "Extract {}/{} frame from {} to {}".format(
                count, connection.msgcount, path_bag, path_video
            )
        )
        print("End extract video from rosbag")
        print("*" * 70)

    @staticmethod
    def rosbag2image(
        path_bag: Path,
        dir_image: Path,
        topic: str,
        typestore: Typestore = get_typestore(Stores.ROS2_HUMBLE),
    ):
        """Extract images from rosbag.

        Args:
            path_bag (Path): The path to save ros bag.
            path_video (Path): The path of folder to save video.
            topic (str): The name of ros topic.
            img_size (tuple): The tuple of (W,H),where W is the weight of video,where H is the height of video.
            fourcc (_type_, optional): The coding mode if video. Defaults to cv2.VideoWriter_fourcc(*"XVID").
            typestore (Typestore, optional): The message types of ros. Defaults to get_typestore(Stores.ROS2_HUMBLE).
            frame_rate (int, optional): The frame rate of video. Defaults to 30.
        """
        print("-" * 70)
        print("Start extract image from rosbag")
        # Create reader instance and open for reading.
        with AnyReader([path_bag], default_typestore=typestore) as reader:
            connections = [x for x in reader.connections if x.topic == topic]
            count = 0
            for connection, timestamp, rawdata in reader.messages(
                connections=connections
            ):
                count += 1
                path_image = dir_image.joinpath("{}.jpg".format(count))
                msg = reader.deserialize(rawdata, connection.msgtype)
                img = message_to_cvimage(msg)
                cv2.imwrite(str(path_image), img)
                # print(
                #     "Extract {}/{} frame from {} to {}".format(
                #         count, connection.msgcount, path_bag, path_video
                #     )
                # )
        print(
            "Extract {}/{} frame from {} to {}".format(
                count, connection.msgcount, path_bag, path_image
            )
        )
        print("End extract image from rosbag")
        print("*" * 70)


if __name__ == "__main__":
    # ---------------------------------------------------------------------------------
    # # rosbag2video
    # paths = [
    #     Path(r"D:\A_Project\database\bus_data\bag\240316\14_20"),
    #     Path(
    #         r"D:\A_Project\database\bus_data\bag\240316\rosbag2_2024_03_16-17_49_22"
    #     ),
    #     Path(r"D:\A_Project\database\bus_data\bag\240316\test_1138"),
    #     Path(
    #         r"D:\A_Project\database\bus_data\bag\240316\t_rosbag2_2024_03_16-12_55_55"
    #     ),
    #     Path(
    #         r"D:\A_Project\database\bus_data\bag\240316\t_rosbag2_2024_03_16-12_58_54"
    #     ),
    #     Path(
    #         r"D:\A_Project\database\bus_data\bag\240316\t_rosbag2_2024_03_16-13_08_34"
    #     ),
    # ]
    # dir_video = Path(r"D:\A_Project\database\bus_data\temp")
    # img_size = (1920, 1536)  # WH
    # topic = "/platform/camera_0/image_compressed"

    # frame_rate = 30
    # typestore = get_typestore(Stores.ROS2_HUMBLE)
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # processor = RosBagProcessor()
    # for path in paths:
    #     path_video = dir_video.joinpath(path.name + ".avi")
    #     processor.rosbag2video(
    #         path, path_video, topic, img_size, fourcc, frame_rate, typestore
    #     )

    # ----------------------------------------------------------------------------------
    # rosbag2image
    paths = [
        Path(
            r"D:\A_Project\database\bus_data\bag\240316\rosbag2_2024_03_16-14_27_51"
        ),
    ]
    dir_save = Path(r"D:\A_Project\database\bus_data\temp")
    topic = "/platform/camera_0/image_compressed"

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    processor = RosBagProcessor()
    for path in paths:
        dir_image = dir_save.joinpath(path.name)
        dir_image.mkdir(parents=True, exist_ok=True)
        processor.rosbag2image(path, dir_image, topic, typestore)
