import os
import moviepy.video.io.ImageSequenceClip
#image_folder='../images/NEWexplosion_u'
image_folders = [
"../images/2compare_mixtureFLIPY_NEW_PINN_mixture_single_source_120000_100_1.0_200_200_800_6_128_tanh_0.07"
]

for top_folder in image_folders:
    print(top_folder)
    folders = [folder for folder in os.listdir(top_folder) if os.path.isdir(os.path.join(top_folder, folder))]
    print(folders)
    for image_folder in folders:
        if image_folder=="x":
            tag = image_folder
            image_folder = top_folder+"/"+image_folder
            print(image_folder)
            fps=1

            image_files = [os.path.join(image_folder,img)
                           for img in os.listdir(image_folder)
                           if img.endswith(".png")]
            print(image_files)
            image_files.sort()
            index_list = []
            for i in image_files:
                time_value = i.split("=")[1]
                time_value = time_value.split(".")[0]
                time_value = float(time_value)
                index_list.append(time_value)
                print(time_value)
            print(index_list)
            sorted_image_files = [x for _, x in sorted(zip(index_list, image_files))]
            print(image_files)
            print(sorted_image_files)
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(sorted_image_files, fps=fps)
            clip.write_videofile('{}_{}.mp4'.format(top_folder.replace("../images/",""),tag))