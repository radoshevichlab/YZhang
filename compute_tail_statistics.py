from kaibu_utils import features_to_mask
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as mpatches
from skimage import morphology
import skan
from skimage.measure import regionprops
import pandas as pd

from imjoy_rpc.hypha import connect_to_server
import asyncio


def compute_tail_length(cell):
    print(np.unique(cell))
    skeleton0 = morphology.skeletonize(cell)
    skel_obj = skan.Skeleton(skeleton0)
    skel_len = skel_obj.path_lengths()
    assert len(skel_len) == 1
    return skel_len[0]


def compute_curvature(coordinates):
    x_t = np.gradient(coordinates[:, 0])
    y_t = np.gradient(coordinates[:, 1])

    vel = np.array([[x_t[i], y_t[i]] for i in range(x_t.size)])
    speed = np.sqrt(x_t * x_t + y_t * y_t)
    tangent = np.array([1 / speed] * 2).transpose() * vel
    ss_t = np.gradient(speed)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)

    curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t) ** 1.5
    return curvature_val.mean()  # remove the first and the last point


def compute_stats(
    df,
    label_image,
    intensity_image,
    plot=False,
    keys=["label", "mean_intensity", "centroid", "bbox"],
    extra_info=None,
):
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(intensity_image)

    props = regionprops(label_image, intensity_image=intensity_image, cache=True)
    for region in props:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        if plot:
            rect = mpatches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor="green",
                linewidth=2,
            )
            ax.add_patch(rect)

        # compute tail length
        skeleton0 = morphology.skeletonize(region.image)
        try:
            # plt.figure()
            # plt.imshow(skeleton0)
            skel_obj = skan.Skeleton(skeleton0)
            lengths = np.array(skel_obj.path_lengths())
            max_index = np.argmax(lengths)
            max_length = np.max(lengths)
            coords = skel_obj.path_coordinates(max_index)
            stats = {
                "curvature": compute_curvature(coords),
                "length": max_length,
                "area": np.sum(region.image),
            }
            for key in keys:
                val = getattr(region, key)
                if isinstance(val, (list, tuple)):
                    for idx, v in enumerate(val):
                        stats[key + "-" + str(idx)] = v
                else:
                    stats[key] = val
            if extra_info:
                stats.update(extra_info)
            df = df.append(stats, ignore_index=True)
        except Exception as exp:
            print("Failed to compute stats: " + str(exp))

    if plot:
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    return df


async def compute_all_stats(lazyio, df, file_path):
    name = os.path.basename(file_path)
    annotation = await lazyio.get_annotation(file_path)
    image = await lazyio.imread(
        file_path, slices=[0, (0, 2), 0, None, None], limit_size=False
    )
    mask_arp3 = np.flipud(features_to_mask(annotation, image.shape[1:], label="arp3"))
    mask_actin = np.flipud(features_to_mask(annotation, image.shape[1:], label="actin"))

    df = compute_stats(
        df,
        mask_actin,
        image[0, :, :],
        plot=False,
        extra_info={"file": name, "channel": "actin"},
    )
    df = compute_stats(
        df,
        mask_arp3,
        image[1, :, :],
        plot=False,
        extra_info={"file": name, "channel": "arp3"},
    )
    return df


async def main():
    api = await connect_to_server({"server_url": "https://ai.imjoy.io"})

    public_services = await api.list_services("public")
    lazyio_service = list(
        filter(lambda service: service["type"] == "lazyio-service", public_services)
    )[-1]
    lazyio = await api.get_service(lazyio_service)
    files = await lazyio.list_dir("/data/arp3-set3")
    image_files = list(filter(lambda x: x["name"].endswith(".czi"), files))

    stats_df = pd.DataFrame(columns=[])

    for idx, file in enumerate(image_files):
        if file.get("annotation") and file.get("type") == "file":
            print(f"Processing {idx+1}/{len(image_files)}", file["name"])
            try:
                stats_df = await compute_all_stats(
                    lazyio, stats_df, f"/data/arp3-set3/{file['name']}"
                )
            except Exception as err:
                print("Failed to process ", file["name"], err)
        else:
            print("Skipping", file["name"])

    stats_df.to_csv("./all_stats_actin_arp3_with_area.csv")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
