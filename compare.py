import cv2
import os
import numpy as np

# Define the directory path
directory_ours = "./Predictions"
Mask_directory = directory_ours + "/Masks"
Position_directory = directory_ours + "/Position"

directory_truth = "./Pred_Official_13"
Mask_Off_directory = directory_truth + "/Masks"
Position_Off_directory = directory_truth + "/Position"

diff_directory = "./Diff_13"
mask_diff_directory = diff_directory + "/Mask"
mask_diff_txt_directory = diff_directory + "/PositionDiff"

class_names = ['drivable', 'alternatives', 'line']

class_names = { index: class_name for index, class_name in enumerate(class_names)}

def Read_Images(mask_dir):
    mask_files = os.listdir(mask_dir)
    # Sort the list of folders in ascending sequence
    sorted_mask_files = sorted(mask_files, key=lambda x: int(os.path.splitext(x)[0]))
    basename = [int(filename.split('.')[0]) for filename in sorted_mask_files]

    Masks_map = {}
    # Iterate over the files
    for file in sorted_mask_files:
        # Make sure it's a JPEG file
        if file.endswith(".jpg"):
            # Read the image
            image_path = os.path.join(mask_dir, file)
            mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
            #masks.append(mask)
            Masks_map[file.split('.')[0]] = mask

    return Masks_map, basename

def Read_Txt(directory, basename):

    data_arrays = []
    for filename in basename:
        filename = str(filename) + '.txt'
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                data = []
                for line in lines:
                    elements = [float(x) for x in line.strip().split()]
                    data.append(elements)
                data_arrays.append(np.array(data))
        except FileNotFoundError:
            zeros = np.zeros((1, 0))
            data_arrays.append(zeros)
            print(f"Error: File '{file_path}' not found in {directory}.")
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    return data_arrays

def calculate_iou_array(array_A, array_B):
    iou_array = np.zeros((array_A.shape[0], array_B.shape[0]))

    for i, box_A in enumerate(array_A):
        for j, box_B in enumerate(array_B):
            # Check if the last two elements are the same
            if np.array_equal(box_A[-1:], box_B[-1:]) and abs(box_A[-2] - box_B[-2]) <= 1e-6:
                # Calculate coordinates of the intersection rectangle
                x1 = max(box_A[0], box_B[0])
                y1 = max(box_A[1], box_B[1])
                x2 = min(box_A[2], box_B[2])
                y2 = min(box_A[3], box_B[3])

                # Calculate area of intersection rectangle
                intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

                # Calculate area of each bounding box
                box_A_area = (box_A[2] - box_A[0] + 1) * (box_A[3] - box_A[1] + 1)
                box_B_area = (box_B[2] - box_B[0] + 1) * (box_B[3] - box_B[1] + 1)

                # Calculate area of union
                union_area = box_A_area + box_B_area - intersection_area

                # Calculate IoU
                iou = intersection_area / union_area

                iou_array[i, j] = iou

    return iou_array

def find_common_elements(A, B):
    common_indices_A = np.zeros(A.shape[0], dtype=bool)
    common_indices_B = np.zeros(B.shape[0], dtype=bool)

    # Check each row in A and B
    for i in range(len(A)):
        for j in range(len(B)):
            # Check if the last two elements are the same and the third last element's difference is not bigger than 1e-6
            if np.allclose(A[i][-3:], B[j][-3:], atol=1e-6) and np.array_equal(A[i][-2:], B[j][-2:]):
                common_indices_A[i] = True
                common_indices_B[j] = True

    common_A = A[common_indices_A]
    common_B = B[common_indices_B]
    
    remain_A = A[~common_indices_A]
    remain_B = B[~common_indices_B]

    return common_A, common_B, remain_A, remain_B

def check_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)
    return image

def generate_colors(i, bgr=False):
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = []
    for iter in hex:
        h = '#' + iter
        palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
    num = len(palette)
    color = palette[int(i) % num]
    return (color[2], color[1], color[0]) if bgr else color

def drawMask(filename, missing, cond):
    color =  (255, 255, 255)
    if cond == 'ours':
        color = ( 0, 255, 0)
    elif cond == 'off':
        color = ( 0, 0, 255)
    elif cond == 'common':
        color = (255, 0 , 0)
    if len(missing) > 0:
        for *xyxy, conf, cls, IoU in missing:
            mask_dir = f'{mask_diff_directory}/{filename}.jpg'
            img_ori = cv2.imread(mask_dir)
            class_num = int(cls)  # integer class
            label = f'{class_names[class_num]} IOU:{IoU:.2f}'
            img_ori = plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 3 * 0.003), 2), xyxy, label, color= color)
            cv2.imwrite(mask_dir, img_ori)

# Read Files from our output and official output
check_directory(diff_directory)

Masks, sorted_basename = Read_Images(Mask_directory)
Positions = Read_Txt(Position_directory, sorted_basename)


Off_Masks, _  = Read_Images(Mask_Off_directory)
Off_Positions = Read_Txt(Position_Off_directory, sorted_basename)


check_directory(mask_diff_directory)
check_directory(mask_diff_txt_directory)

for key, image in Off_Masks.items():
    check_directory(f'{mask_diff_directory}')
    rgb_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    if key in Masks:
        OverLap = cv2.bitwise_and(Masks[key], Off_Masks[key])
        Offset_Mask = Masks[key] - OverLap
        Offset_Off = Off_Masks[key] - OverLap

        rgb_image[:, :, 2] = Offset_Off 
        rgb_image[:, :, 1] = Offset_Mask

        cv2.imwrite(f"{mask_diff_directory}/{key}.jpg", rgb_image)
    else:
        rgb_image[:, :, 2] = image  
        cv2.imwrite(f"{mask_diff_directory}/{key}.jpg", rgb_image)
        print('==========================================') 
        print(f'Image: {key}.jpg not found in our prediction.')
        print('==========================================')

for off_pos, pos, filename in zip(Off_Positions, Positions, sorted_basename):
    print(f'===={filename}====')
    # Calculate IOU
    IoU = calculate_iou_array(off_pos, pos) # 12 * 11
    common_off, common_ours, remain_off, remain_our = [], [], [], []
    if IoU.shape[0] != 0 and IoU.shape[1] != 0:
        max_elements_row = np.amax(IoU, axis=1)  # Maximum elements along axis 1 (rows)
        max_elements_col = np.amax(IoU, axis=0)  # Max Column

        bound = 0.98
        missing_indices_row = np.where(max_elements_row < bound)[0]
        missing_indices_col = np.where(max_elements_col < bound)[0]

        print(f'Missing Off indices {missing_indices_row}')
        print(f'Missing Ours indices {missing_indices_col}')

        missing_officials = off_pos[missing_indices_row]
        if len(missing_officials) > 0:
            missing_IOU_off = max_elements_row[missing_indices_row]
            missing_officials = np.concatenate((missing_officials,  np.expand_dims(missing_IOU_off, axis=1)), axis=1)
        else:
            missing_officials = np.empty((0, 7))

        missing_ours = pos[missing_indices_col]
        if len(missing_ours) > 0:
            missing_IOU_our = max_elements_col[missing_indices_col]
            missing_ours = np.concatenate((missing_ours, np.expand_dims(missing_IOU_our, axis=1)), axis=1)
        else:
            missing_ours = np.empty((0, 7))
        
        common_off, common_ours, remain_off, remain_our = find_common_elements(missing_officials, missing_ours)

    elif IoU.shape[0] != 0 or IoU.shape[1] != 0:
        if IoU.shape[0] == 0:
            scalar_array = np.full((pos.shape[0], 1), 0.0)
            remain_our = np.concatenate((pos, scalar_array), axis=1)
            print('Remain Ours:')
            print(remain_our)
        if IoU.shape[1] == 0: 
            scalar_array = np.full((off_pos.shape[0], 1), 0.0)
            remain_off = np.concatenate((off_pos, scalar_array), axis=1)
            print('Remain Official:')
            print(remain_off)

    drawMask(filename, common_off, 'common')
    drawMask(filename, remain_off, 'off')
    drawMask(filename, remain_our, 'ours')

    with open(f'{mask_diff_txt_directory}/{filename}.txt', 'w') as file:

        for rowA, rowB in zip(common_off, common_ours):
            file.write(' '.join(map(str, rowA)) + '\n')
            file.write(' '.join(map(str, rowB)) + '\n')
        
        file.write("===\n")    

        for row in remain_off:
            file.write(' '.join(map(str, row)) + '\n')

        # Add "===" between arrays
        file.write("===\n")

        for row in remain_our:
            file.write(' '.join(map(str, row))  + '\n')
    
print(f"Store Compared Results into {diff_directory}")