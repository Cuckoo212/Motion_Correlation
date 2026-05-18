"""
Generate qualitative motion segmentation and correlation comparison figures.
The figures consist of color images with overlaid segmentation masks.
This requires having pre-computed results saved on the disk.
"""

import os
import pathlib
import json
from copy import deepcopy

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tqdm

from cmcor import motion_perception_cli
from cmcor import motion_perception_accuracy
from cmcor import motion_perception_multi_eval
from cmcor import motion_perception_settings
from cmcor import connect_segment




def load_rgb_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert(image is not None)
    image = image[...,::-1]
    return image


def crop_image(image, crop_origin, crop_size):
    if crop_origin is not None:
        image = image[
            crop_origin[1]:crop_origin[1]+crop_size[1],
            crop_origin[0]:crop_origin[0]+crop_size[0], ...]
    return image


def rgb_average(image_root, crop_origin, crop_size):
    average_image = None
    image_count = 0
    for name in sorted(os.listdir(image_root)):
        if not name.startswith("rgb_"):
            continue
        image_path = os.path.join(image_root, name)
        img = load_rgb_image(image_path)
        img = crop_image(img, crop_origin, crop_size)
        img = img.astype(np.float32)
        if average_image is None:
            average_image = img
            image_count = 1
        else:
            average_image = average_image + img
            image_count += 1
    assert(image_count > 0)
    average_image = average_image/image_count
    average_image[average_image < 0] = 0
    average_image[average_image > 255] = 255
    average_image = average_image.astype(np.uint8)
    return average_image


def load_grasps(output_root):
    grasp_mask = None
    grasp_idx = 0
    grasp_path = os.path.join(output_root, f"grasped_{grasp_idx}.png")
    while os.path.isfile(grasp_path):
        image = cv2.imread(grasp_path, cv2.IMREAD_UNCHANGED)
        assert(image is not None)
        if grasp_mask is None:
            grasp_mask = image > 0
        else:
            grasp_mask = np.logical_or(grasp_mask, image > 0)
        grasp_idx += 1
        grasp_path = os.path.join(output_root, f"grasped_{grasp_idx}.png")
    return grasp_mask


class ComparisonPlots():
    def __init__(self, max_votes=2):
        self.max_votes = max_votes
        self.cmap = matplotlib.cm.get_cmap('Greens')

    def colorbar(self, ax):
        votes = [vote for vote in range(1, self.max_votes+1)]
        votes_str = [str(vote) for vote in votes]
        vote_colors = [
            list(self.cmap(vote/self.max_votes)[0:3]) #[0, int((255.0/self.max_votes)*vote), 0]
            for vote in votes]
        vote_colors = vote_colors[::-1]
        vote_colors = np.array(vote_colors)
        vote_colors = vote_colors*255
        vote_colors = vote_colors.tolist()
        vote_image = np.array([vote_colors], dtype=np.uint8)
        vote_image = np.transpose(vote_image,(1,0,2))
        ax.imshow(vote_image, extent=[0,1,0.5,self.max_votes+0.5])
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        ax.set_title("Votes")

    def rgb_vote_image(self, target_rgb, vote_image):
        rgb_votes = deepcopy(target_rgb)
        vote_image[vote_image>self.max_votes] = self.max_votes
        some_votes = vote_image>0
        vote_colors = [
            list(self.cmap(vote/self.max_votes)[0:3])
            #[0, int((255.0/self.max_votes)*vote), 0]
            for vote in range(1, self.max_votes+1)]
        vote_colors = np.array(vote_colors)
        vote_colors = vote_colors*255
        vote_colors = vote_colors.astype(np.uint8)
        
        rgb_votes[some_votes,:] = vote_colors[vote_image[some_votes]-1,:]
        #rgb_votes[some_votes,1] = vote_image[some_votes]*(255.0/self.max_votes)
        #rgb_votes[some_votes,0] = 0
        #rgb_votes[some_votes,2] = 0
        return rgb_votes

    def rgb_grasp_image(self, target_rgb, grasp_mask):
        rgb_grasps = deepcopy(target_rgb)
        rgb_grasps[grasp_mask,1] = 255
        rgb_grasps[grasp_mask,0] = 0
        rgb_grasps[grasp_mask,2] = 0
        return rgb_grasps

    def create_plots(
            self, folder_comparison, folder_gt,
            folder_correlation, folder_segmentation, data_root,
            param_corr, param_seg):
        pathlib.Path(folder_comparison).mkdir(parents=True, exist_ok=True)
        for name in tqdm.tqdm(sorted(os.listdir(folder_gt))):
            if not name.startswith("20"):
                continue
            results_corr_path = os.path.join(folder_correlation, name)
            if not os.path.isdir(results_corr_path):
                continue
            results_seg_path = os.path.join(folder_segmentation, name)
            if not os.path.isdir(results_seg_path):
                continue
            gt_path = os.path.join(folder_gt, name)
            gt_images = sorted(os.listdir(gt_path))
            if not gt_images:
                continue
            gt_img_path = os.path.join(gt_path, gt_images[0])
            gt_image = cv2.imread(gt_img_path, cv2.IMREAD_UNCHANGED)
            gt_mask = gt_image > 0
            sequence_codes = motion_perception_cli.load_sequence_action_codes(
                results_corr_path)
            vote_image_corr = motion_perception_accuracy.load_predictions(
                results_corr_path, sequence_codes, param_corr,
                "Motion correlation")
            vote_image_seg = motion_perception_accuracy.load_predictions(
                results_seg_path, sequence_codes, param_seg,
                "Motion segmentation")
            json_path = os.path.join(data_root, name, "actions_gripper.json")
            with open(json_path, "r") as fp:
                json_dict = json.load(fp)
            if "crop_origin" in json_dict and "crop_size" in json_dict:
                crop_origin = json_dict["crop_origin"]
                crop_size = json_dict["crop_size"]
            else:
                crop_origin = None
                crop_size = None
            vote_image_corr = crop_image(
                vote_image_corr, crop_origin, crop_size)
            vote_image_seg = crop_image(
                vote_image_seg, crop_origin, crop_size)
            gt_mask = crop_image(
                gt_mask, crop_origin, crop_size)
            rgb_avg = rgb_average(
                os.path.join(data_root, name), crop_origin, crop_size)
            grasp_mask = load_grasps(results_corr_path)
            grasp_mask = crop_image(
                grasp_mask, crop_origin, crop_size)
            target_image_path = os.path.join(
                results_corr_path, "target_rgb.png")
            target_image = load_rgb_image(target_image_path)
            target_image = crop_image(
                target_image, crop_origin, crop_size)
            plt.rcParams['font.family'] = "DejaVu Sans" #"FreeSerif" #"Times New Roman"
            plt.rcParams['font.size'] = 16
            fig, axs = plt.subplots(
                1,4, gridspec_kw={"width_ratios":[0.323,0.323,0.323,0.05]})
            fig.set_size_inches(10.5,4.28)
            rgb_grasps = self.rgb_grasp_image(rgb_avg, grasp_mask)
            axs[0].imshow(rgb_grasps)
            axs[0].set_title("Grasps")
            axs[1].imshow(self.rgb_vote_image(target_image, vote_image_seg))
            axs[1].set_title("MSeg")
            axs[2].imshow(self.rgb_vote_image(target_image, vote_image_corr))
            axs[2].set_title("MCor (ours)")
            self.colorbar(axs[3])
            for ax in axs[0:-1]:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.tight_layout()
            save_path = os.path.join(folder_comparison, name + ".png")
            plt.savefig(save_path)
            save_path = os.path.join(folder_comparison, name + ".pdf")
            plt.savefig(save_path)

    def create_plots_multi_grasp_video(
            self, folder_video, folder_gt,
            folder_results, folder_results_single, data_root,
            sequences, parameters, method_title):
        pathlib.Path(folder_video).mkdir(parents=True, exist_ok=True)

        for seq, param in tqdm.tqdm(
                zip(sequences, parameters), total=len(sequences)):
            multi_name = motion_perception_multi_eval.get_multi_name(seq)
            results_path = os.path.join(folder_results, multi_name)
            if not os.path.isdir(results_path):
                continue
            gt_path = os.path.join(folder_gt, seq[-1])
            gt_images = sorted(os.listdir(gt_path))
            if not gt_images:
                continue
            sequence_codes = motion_perception_cli.load_sequence_action_codes(
                results_path)
            vote_images = motion_perception_accuracy.load_predictions_multi_grasp(
                results_path, sequence_codes, param, method_title)
            grasp_mask_accumulated = None
            for grasp_number, clip_name in enumerate(seq):
                clip_path = os.path.join(folder_results_single, clip_name)
                sequence_codes_clip = motion_perception_cli.load_sequence_action_codes(
                    clip_path)
                vote_image_clip = motion_perception_accuracy.load_predictions(
                    clip_path, sequence_codes_clip, param, method_title)
                json_path = os.path.join(
                    data_root, clip_name, "actions_gripper.json")
                with open(json_path, "r") as fp:
                    json_dict = json.load(fp)
                crop_origin = None
                crop_size = None
                if "crop_origin" in json_dict and "crop_size" in json_dict:
                    crop_origin = json_dict["crop_origin"]
                    crop_size = json_dict["crop_size"]
                    vote_image_clip = vote_image_clip[
                        crop_origin[1]:crop_origin[1]+crop_size[1],
                        crop_origin[0]:crop_origin[0]+crop_size[0]]
                target_image_clip = load_rgb_image(
                    os.path.join(clip_path, "target_rgb.png"))
                target_image_clip = crop_image(
                    target_image_clip, crop_origin, crop_size)
                rgb_avg = rgb_average(
                    os.path.join(data_root, clip_name), crop_origin, crop_size)
                grasp_mask = load_grasps(clip_path)
                grasp_mask = crop_image(grasp_mask, crop_origin, crop_size)
                if grasp_mask_accumulated is None:
                    grasp_mask_accumulated = grasp_mask
                else:
                    grasp_mask_accumulated = np.logical_or(
                        grasp_mask_accumulated, grasp_mask)
                plt.rcParams['font.family'] = "DejaVu Sans" #"FreeSerif" #"Times New Roman"
                plt.rcParams['font.size'] = 16
                fig, axs = plt.subplots(
                    1,4, gridspec_kw={"width_ratios":[0.323,0.323,0.323,0.05]})
                image_height = target_image_clip.shape[0]
                image_width = target_image_clip.shape[1]
                default_dpi = 100
                #plot_height = (image_height*(428/368))/default_dpi
                plot_scale = 428/image_height
                plotted_image_width = image_width*plot_scale
                plot_width = (
                    (plotted_image_width*3 + 24.25*2 + 52.5)/default_dpi)
                fig.set_size_inches(plot_width, 4.28) # 10.5, 4.28
                rgb_grasps = self.rgb_grasp_image(
                    rgb_avg, grasp_mask_accumulated)
                axs[0].imshow(rgb_grasps)
                axs[0].set_title("Grasps")
                axs[1].imshow(
                    self.rgb_vote_image(target_image_clip, vote_image_clip))
                axs[1].set_title(f"Grasp {grasp_number+1}")
                self.colorbar(axs[3])
                colorbar_ticks = axs[3].get_yticks()
                colorbar_labels = [f"{int(tick)}" for tick in colorbar_ticks]
                assert(colorbar_labels[-2] == f"{self.max_votes}")
                colorbar_labels[-2] = f"{self.max_votes}+"
                colorbar_labels = colorbar_labels[1:-1]
                colorbar_ticks = colorbar_ticks[1:-1]
                axs[3].set_yticks(colorbar_ticks, colorbar_labels)
                for ax in axs[0:-1]:
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                axs[2].axis('off')
                plt.tight_layout()
                save_path = os.path.join(
                    folder_video, multi_name + f"_{grasp_number}.png")
                plt.savefig(save_path)
                if grasp_number + 1 < len(seq):
                    clip_path = os.path.join(
                        folder_results_single, seq[grasp_number + 1])
                    grasp_mask = load_grasps(clip_path)
                    grasp_mask = crop_image(grasp_mask, crop_origin, crop_size)
                    next_grasp = np.logical_or(
                        grasp_mask_accumulated, grasp_mask)
                    rgb_grasps = self.rgb_grasp_image(
                        rgb_avg, next_grasp)
                    axs[0].imshow(rgb_grasps)
                    save_path = os.path.join(
                        folder_video, multi_name + f"_{grasp_number}_next.png")
                    plt.savefig(save_path)


    def create_plots_multi_grasp(
            self, folder_comparison,
            folder_results, folder_results_single, data_root,
            sequences, parameters, method_title, plot_splines=True):
        pathlib.Path(folder_comparison).mkdir(parents=True, exist_ok=True)

        for seq, param in tqdm.tqdm(
                zip(sequences, parameters), total=len(sequences)):
            multi_name = motion_perception_multi_eval.get_multi_name(seq)
            results_path = os.path.join(folder_results, multi_name)
            if not os.path.isdir(results_path):
                continue
            sequence_codes = motion_perception_cli.load_sequence_action_codes(
                results_path)
            vote_images = motion_perception_accuracy.load_predictions_multi_grasp(
                results_path, sequence_codes, param, method_title)

            first_grasp_path = os.path.join(folder_results_single, seq[0])
            sequence_codes_first = motion_perception_cli.load_sequence_action_codes(
                first_grasp_path)
            vote_image_first = motion_perception_accuracy.load_predictions(
                first_grasp_path, sequence_codes_first, param, method_title)
            target_image_first = load_rgb_image(
                os.path.join(first_grasp_path, "target_rgb.png"))

            json_path = os.path.join(
                data_root, seq[-1], "actions_gripper.json")
            with open(json_path, "r") as fp:
                json_dict = json.load(fp)
            crop_origin = None
            crop_size = None
            if "crop_origin" in json_dict and "crop_size" in json_dict:
                crop_origin = json_dict["crop_origin"]
                crop_size = json_dict["crop_size"]
                for image_idx in range(len(vote_images)):
                    vote_images[image_idx] = vote_images[image_idx][
                        crop_origin[1]:crop_origin[1]+crop_size[1],
                        crop_origin[0]:crop_origin[0]+crop_size[0]]
                vote_image_first = vote_image_first[
                    crop_origin[1]:crop_origin[1]+crop_size[1],
                    crop_origin[0]:crop_origin[0]+crop_size[0]]
            target_image_first = crop_image(
                target_image_first, crop_origin, crop_size)

            rgb_avg = rgb_average(
                os.path.join(data_root, seq[0]), crop_origin, crop_size)
            if len(seq) > 1:
                rgb_avg = rgb_avg.astype(int)
                for name in seq[1:]:
                    avg = rgb_average(
                        os.path.join(data_root, name), crop_origin, crop_size)
                    rgb_avg += avg
                rgb_avg = rgb_avg / len(seq)
                rgb_avg[rgb_avg < 0] = 0
                rgb_avg[rgb_avg > 255] = 255
                rgb_avg = rgb_avg.astype(np.uint8)

            grasp_mask = load_grasps(results_path)
            grasp_mask = crop_image(
                grasp_mask, crop_origin, crop_size)
            target_image_path = os.path.join(
                results_path, "target_rgb.png")
            target_image = load_rgb_image(target_image_path)
            target_image = crop_image(
                target_image, crop_origin, crop_size)

            plt.rcParams['font.family'] = "DejaVu Sans" #"FreeSerif" #"Times New Roman"
            plt.rcParams['font.size'] = 16
            #fig, axs = plt.subplots(
            #    1,4, gridspec_kw={"width_ratios":[0.323,0.323,0.323,0.05]})
            fig, axs = plt.subplots(
                1,5, gridspec_kw={"width_ratios":[0.242,0.242,0.242,0.242,0.05]})
            image_height = target_image.shape[0]
            image_width = target_image.shape[1]
            default_dpi = 100
            #plot_height = (image_height*(428/368))/default_dpi
            plot_scale = 428/image_height
            plotted_image_width = image_width*plot_scale
            #plot_width = (
            #    (plotted_image_width*3 + 24.25*2 + 52.5)/default_dpi)
            plot_width = (
                (plotted_image_width*4 + 24.25*2 + 52.5)/default_dpi)
            fig.set_size_inches(plot_width, 4.28) # 10.5, 4.28
            rgb_grasps = self.rgb_grasp_image(rgb_avg, grasp_mask)
            axs[0].imshow(rgb_grasps)
            axs[0].set_title("Grasps")
            axs[1].imshow(
                self.rgb_vote_image(target_image_first, vote_image_first))
            axs[1].set_title("First Grasp")
            axs[2].imshow(self.rgb_vote_image(target_image, vote_images[-1]))
            axs[2].set_title(f"All Grasps ({len(vote_images)})")
            axs[3].imshow(self.rgb_vote_image(target_image, vote_images[-1]))
            if plot_splines:
                spline_points = connect_segment.process_mask_with_reordering(
                    vote_images[-1] >= 2)
                if len(spline_points) > 0:
                    axs[3].plot(
                        spline_points[:, 0], spline_points[:, 1], 'r-', lw=2)
            axs[3].set_title(f"All Grasps ({len(vote_images)})")
            self.colorbar(axs[4])
            colorbar_ticks = axs[4].get_yticks()
            colorbar_labels = [f"{int(tick)}" for tick in colorbar_ticks]
            assert(colorbar_labels[-2] == f"{self.max_votes}")
            colorbar_labels[-2] = f"{self.max_votes}+"
            colorbar_labels = colorbar_labels[1:-1]
            colorbar_ticks = colorbar_ticks[1:-1]
            axs[4].set_yticks(colorbar_ticks, colorbar_labels)
            for ax in axs[0:-1]:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.tight_layout()
            save_path = os.path.join(folder_comparison, multi_name + ".pdf")
            plt.savefig(save_path)
            save_path = os.path.join(folder_comparison, multi_name + ".png")
            plt.savefig(save_path)
            # Extra outputs:
            folder_extra = os.path.join(folder_comparison, "extra")
            pathlib.Path(folder_extra).mkdir(parents=True, exist_ok=True)
            all_grasps_at_2 = 255*(vote_images[-1] >= 2).astype(np.uint8)
            save_path = os.path.join(
                folder_extra, multi_name + "_all_grasps_at_2.png")
            cv2.imwrite(save_path, all_grasps_at_2)
            all_grasps_at_1 = 255*(vote_images[-1] >= 1).astype(np.uint8)
            save_path = os.path.join(
                folder_extra, multi_name + "_all_grasps_at_1.png")
            cv2.imwrite(save_path, all_grasps_at_1)
            save_path = os.path.join(
                folder_extra, multi_name + "_target_cropped.png")
            cv2.imwrite(save_path, target_image[...,::-1])





def single_grasp_comparison():
    mp_settings = motion_perception_settings.MotionPerceptionSettings()
    gt_root = mp_settings.gt_root
    out_root = mp_settings.output_root
    data_root = mp_settings.data_root
    folder_out_correlation = os.path.join(out_root, "correlation_masks")
    folder_out_segmentation = os.path.join(out_root, "segmentation_masks")
    gt_root_validation = os.path.join(gt_root, "validation")
    gt_root_test = os.path.join(gt_root, "test")
    all_betas = mp_settings.eval_all_betas
    params_seg = mp_settings.eval_params_segmentation
    params_corr = mp_settings.eval_params_correlation
    params_flow_std = mp_settings.eval_params_flow_std

    folder_comparison = os.path.join(
        out_root, "comparison_plots")
    param_index = all_betas.index(mp_settings.beta_to_plot)
    p_corr = params_corr[param_index]  #[86, 18]
    p_seg = params_seg[param_index] #47
    plots = ComparisonPlots(max_votes=2)
    print("Single grasp validation set...")
    plots.create_plots(
        folder_comparison, gt_root_validation,
        folder_out_correlation, folder_out_segmentation,
        data_root, p_corr, p_seg)
    print("Single grasp test set...")
    plots.create_plots(
        folder_comparison, gt_root_test,
        folder_out_correlation, folder_out_segmentation,
        data_root, p_corr, p_seg)


def double_grasp_comparison():
    mp_settings = motion_perception_settings.MotionPerceptionSettings()
    gt_root = mp_settings.gt_root
    out_root = mp_settings.output_root
    data_root = mp_settings.data_root
    folder_out_correlation = os.path.join(out_root, "correlation_masks_multi")
    folder_results_single = os.path.join(out_root, "correlation_masks")
    gt_root_test = os.path.join(gt_root, "test")
    sequences = mp_settings.double_grasp_sequences
    params = mp_settings.double_grasp_params

    folder_comparison = os.path.join(
        out_root, "comparison_plots_multi")
    plots = ComparisonPlots(max_votes=4)
    print("Double grasp test set...")
    plots.create_plots_multi_grasp(
        folder_comparison,
        folder_out_correlation, folder_results_single, data_root, sequences,
        params, "Motion correlation")


def video_multigrasp():
    mp_settings = motion_perception_settings.MotionPerceptionSettings()
    gt_root = mp_settings.gt_root
    out_root = mp_settings.output_root
    data_root = mp_settings.data_root
    folder_out_correlation = os.path.join(out_root, "correlation_masks_multi")
    folder_results_single = os.path.join(out_root, "correlation_masks")
    gt_root_test = os.path.join(gt_root, "test")
    sequences = mp_settings.double_grasp_sequences
    params = mp_settings.double_grasp_params

    folder_video = os.path.join(
        out_root, "video_plots_multi")
    plots = ComparisonPlots(max_votes=4)
    print("Multi-grasp video plots...")
    sequences = sequences[-6:]
    params = params[-6:]
    plots.create_plots_multi_grasp_video(
        folder_video, gt_root_test, folder_out_correlation,
        folder_results_single,
        data_root, sequences, params, "Motion correlation")


if __name__ == "__main__":
    single_grasp_comparison()
    double_grasp_comparison()
    #video_multigrasp()
