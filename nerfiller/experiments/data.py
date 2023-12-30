"""We keep track of data here."""

dataset_name_to_prompt = {
    "chair": "a photo of a chair",
    "drums": "a photo of drums",
    "ficus": "a photo of a ficus plant",
    "hotdog": "a photo of a hotdog",
    "lego": "a photo of a lego bulldozer",
    "materials": "a photo of materials",
    "mic": "a photo of a microphone",
    "ship": "a photo of a ship",
}

blender_dataset_names = set(["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"])
synthetic_dataset_names = set(["dumptruck", "boot", "bear", "bearface", "bearears", "cat", "turtle"]).union(
    blender_dataset_names
)
synthetic_black_background_dataset_names = set(["cat", "turtle"])  # otherwise white
occluder_dataset_names = set(
    [
        "billiards",
        "dumptruck",
        "office",
        "backpack",
        "drawing",
        "norway",
        "boot",
        "bear",
        "cat",
        "turtle",
    ]
)
# "bearface" and "bearears" exist too as reference inpaint examples
nerfbusters_dataset_names = set(["aloe", "table", "tinytable", "couch"])
forward_facing_dataset_names = set(["backpack"])
turn_off_view_dependence_dataset_names = set(["billiards-view-ind"])
forward_facing_snippet = "--pipeline.model.distortion_loss_mult 0.0"
turn_off_view_dependence_snippet = (
    "--pipeline.model.use_direction_encoding False --pipeline.model.use_appearance_embedding False"
)

eval_all_dataset_names = set()
eval_all_dataset_names = eval_all_dataset_names.union(occluder_dataset_names)
eval_all_dataset_names = eval_all_dataset_names.union(nerfbusters_dataset_names)

dataset_render_near_plane = 0.5  # used for all dataset renderings
camera_path_inference_near_planes = {}  # if specified, use for the camera path videos. otherwise use default of 0.0
camera_path_inference_near_planes = {"norway": 0.5}
camera_path_inference_near_planes = {"dumptruck": 0.5}
camera_path_inference_near_planes = {"backpack": 0.5}
camera_path_inference_near_planes = {"boot": 0.5}
camera_path_inference_near_planes = {"turtle": 0.25}

dataset_name_method_modifications = {
    "bearface+grid-prior-du-reference": '--pipeline.text_guidance_scale 15.0 --pipeline.prompt "black bear" --pipeline.lower-bound 0.8',
    "bearears+grid-prior-du-reference": '--pipeline.text_guidance_scale 15.0 --pipeline.prompt "bunny ears" --pipeline.lower-bound 0.8',
    "cat+grid-prior-du-reference": '--pipeline.text_guidance_scale 15.0 --pipeline.prompt "santa claus" --pipeline.lower-bound 0.8',
}

# 100 randomly sampled camera pairs to evaluate the novel view camera path
# these values * the number of frames in the video to get the video frame index
novel_view_pairs = [
    (0.5066666666666667, 0.68),
    (0.05, 0.7366666666666667),
    (0.6066666666666667, 0.63),
    (0.25, 0.7433333333333333),
    (0.37666666666666665, 0.49),
    (0.29, 0.36666666666666664),
    (0.43666666666666665, 0.7333333333333333),
    (0.5366666666666666, 0.5966666666666667),
    (0.6366666666666667, 0.94),
    (0.5033333333333333, 0.56),
    (0.6466666666666666, 0.9066666666666666),
    (0.5333333333333333, 0.72),
    (0.12333333333333334, 0.6066666666666667),
    (0.35, 0.6),
    (0.43666666666666665, 0.89),
    (0.07, 0.09666666666666666),
    (0.7233333333333334, 0.98),
    (0.11333333333333333, 0.24666666666666667),
    (0.16666666666666666, 0.5433333333333333),
    (0.14333333333333334, 0.19),
    (0.05333333333333334, 0.96),
    (0.7366666666666667, 0.93),
    (0.38, 0.56),
    (0.01, 0.12666666666666668),
    (0.37666666666666665, 0.61),
    (0.29, 0.65),
    (0.24, 0.6566666666666666),
    (0.49, 0.77),
    (0.6133333333333333, 0.7366666666666667),
    (0.08666666666666667, 0.8666666666666667),
    (0.5433333333333333, 0.8466666666666667),
    (0.4033333333333333, 0.5333333333333333),
    (0.27, 0.56),
    (0.5466666666666666, 0.9866666666666667),
    (0.18, 0.47),
    (0.4633333333333333, 0.87),
    (0.38333333333333336, 0.46),
    (0.54, 0.6266666666666667),
    (0.21666666666666667, 0.26666666666666666),
    (0.14, 0.8266666666666667),
    (0.08, 0.9566666666666667),
    (0.18, 0.94),
    (0.03333333333333333, 0.8633333333333333),
    (0.023333333333333334, 0.93),
    (0.13333333333333333, 0.8166666666666667),
    (0.1, 0.53),
    (0.5366666666666666, 0.63),
    (0.043333333333333335, 0.22),
    (0.27, 0.91),
    (0.30333333333333334, 0.6866666666666666),
    (0.31333333333333335, 0.7233333333333334),
    (0.2733333333333333, 0.5266666666666666),
    (0.12666666666666668, 0.7266666666666667),
    (0.55, 0.92),
    (0.056666666666666664, 0.29),
    (0.023333333333333334, 0.31),
    (0.01, 0.05),
    (0.05333333333333334, 0.9933333333333333),
    (0.09666666666666666, 0.74),
    (0.4066666666666667, 0.8933333333333333),
    (0.55, 0.9766666666666667),
    (0.08333333333333333, 0.5833333333333334),
    (0.29, 0.6833333333333333),
    (0.04, 0.67),
    (0.05, 0.9633333333333334),
    (0.35333333333333333, 0.54),
    (0.36, 0.78),
    (0.013333333333333334, 0.07),
    (0.4866666666666667, 0.5833333333333334),
    (0.12333333333333334, 0.8333333333333334),
    (0.15333333333333332, 0.45666666666666667),
    (0.27, 0.4166666666666667),
    (0.1, 0.86),
    (0.17666666666666667, 0.21666666666666667),
    (0.77, 0.9666666666666667),
    (0.07, 0.42333333333333334),
    (0.5733333333333334, 0.77),
    (0.5633333333333334, 0.67),
    (0.09666666666666666, 0.41333333333333333),
    (0.36, 0.64),
    (0.043333333333333335, 0.7733333333333333),
    (0.14, 0.7666666666666667),
    (0.6233333333333333, 0.64),
    (0.02, 0.7666666666666667),
    (0.03666666666666667, 0.6333333333333333),
    (0.07333333333333333, 0.55),
    (0.23, 0.7333333333333333),
    (0.11666666666666667, 0.73),
    (0.7433333333333333, 0.94),
    (0.49666666666666665, 0.95),
    (0.19333333333333333, 0.22333333333333333),
    (0.43333333333333335, 0.7233333333333334),
    (0.13333333333333333, 0.47),
    (0.09, 0.6633333333333333),
    (0.023333333333333334, 0.15666666666666668),
    (0.006666666666666667, 0.93),
    (0.16333333333333333, 0.38666666666666666),
    (0.6833333333333333, 0.9133333333333333),
    (0.16333333333333333, 0.8866666666666667),
    (0.016666666666666666, 0.06333333333333334),
]
