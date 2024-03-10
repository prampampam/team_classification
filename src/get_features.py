import json
from PIL import Image
import numpy as np
import pandas as pd


def get_image_features(mode="train"):
    if mode == "train":
        with open('../data/bboxes.json') as f:
            meta_data = json.load(f)
    else:
        with open('../data/test/test_bboxes.json') as f:
            meta_data = json.load(f)

    df = pd.DataFrame()

    for img_id, _ in meta_data.items():
        print(img_id)
        img_id = int(img_id)
        if mode == "train":
            img = Image.open(f'../data/frames/{img_id}.jpeg')
        else:
            img = Image.open(f'../data/test/frames/{img_id}.jpeg')
        width, height = img.size

        result = []
        for player_id in range(10):
            features = dict()
            features["img_id"] = img_id
            features["player_id"] = player_id

            if mode == "train":
                features["team"] = meta_data[str(img_id)][str(player_id)]["team"]
            x, y, w, h = meta_data[str(img_id)][str(player_id)]["box"]

            x_up = x * width
            y_up = y * height
            x_down = (x + w) * width
            y_down = (y + h) * height
            img_crop = img.crop((x_up, y_up, x_down, y_down))

            # TODO: попробовать брать не весь bbox, а только центральную часть.
            img_crop_rgb = img_crop.convert("RGB")
            r, g, b = img_crop_rgb.split()
            features["r_mean"] = np.array(r).mean()
            features["g_mean"] = np.array(g).mean()
            features["b_mean"] = np.array(b).mean()

            img_crop_hsv = img_crop.convert("HSV")
            h, s, v = img_crop_hsv.split()
            features["h_mean"] = np.array(h).mean()
            features["s_mean"] = np.array(s).mean()
            features["v_mean"] = np.array(v).mean()

            img_crop_array = np.array(img_crop)
            tmp_df = pd.DataFrame(img_crop_array[:, :, 0])
            tmp_res = []
            for i in range(0, 256):
                tmp_res.append((tmp_df == i).sum().sum())
            tmp_res = pd.Series(tmp_res)

            for step in [4, 16, 32, 64]:
                n_part = 0
                for i in range(0, 256, step):
                    ind_start = i
                    ind_end = i + step
                    if ind_end < 256:
                        features[f'mean_{step}_{n_part}'] = tmp_res.iloc[ind_start:ind_end].mean() / tmp_df.size
                        n_part += 1
            result.append(features)

        df = pd.concat([df, pd.DataFrame(result)])

    df.to_parquet(f'../data/df_features_{mode}.pa')


if __name__ == "__main__":
    get_image_features("train")
    get_image_features("test")
