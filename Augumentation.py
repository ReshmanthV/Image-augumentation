import cv2
import pybboxes as pbx
import uuid
import os


class ImageAugmentation:
    def rotate_bb_90_deg_clockwise(self, bndbox,
                                   img_width):  # just passing width of image is enough for 90-degree rotation.
        # x_min,y_min,x_max,y_max = bndbox
        x_min, y_min, x_max, y_max = bndbox
        new_xmin = img_width - y_max  # Reflection about center X-line
        new_ymin = x_min
        new_xmax = img_width - y_min  # Reflection about center X-line
        new_ymax = x_max
        return [new_xmin, new_ymin, new_xmax, new_ymax]

    def rotate_bb_90_deg_counter_clockwise(bndbox,
                                           img_width):  # just passing width of image is enough for 90-degree rotation.
        x_min, y_min, x_max, y_max = bndbox
        new_xmin = y_min
        new_ymin = img_width - x_max
        new_xmax = y_max
        new_ymax = img_width - x_min
        return [new_xmin, new_ymin, new_xmax, new_ymax]

    @staticmethod
    def flip_horizontal(bb, img):
        new_coordinates_list = []
        actual_height, actual_width, _ = img.shape
        image_hor = cv2.flip(img, 1)
        for each_line in bb:
            act_x1, act_y1, act_x2, act_y2 = each_line
            new_x1 = actual_width - act_x1
            new_x2 = actual_width - act_x2
            new_coordinates_list.append([new_x2, act_y1, new_x1, act_y2])
        return new_coordinates_list, image_hor

    @staticmethod
    def vertical_flip_image(bb, img):
        coordinates_list = []
        height, width, _ = img.shape
        vertical_flip = cv2.flip(img, 0)
        for each_line in bb:
            act_x1, act_y1, act_x2, act_y2 = each_line
            new_y1 = height - act_y1
            new_y2 = height - act_y2
            coordinates_list.append([act_x1, new_y2, act_x2, new_y1])
        return coordinates_list, vertical_flip

    @staticmethod
    def rotate_90_deg_clockwise(bb, img):
        augmented_coordinates = []
        img_cw_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_width = img_cw_90.shape[1]
        for bounding_box in bb:
            augmented_bb = rotate90Deg_clockwise(bounding_box, img_width)
            augmented_coordinates.append(augmented_bb)
        return augmented_coordinates, img_cw_90

    @staticmethod
    def rotate_90_deg_counter_clockwise(bb, img):
        img_width = img.shape[1]
        augmented_coordinates = []
        img_ccw_90 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        for bounding_box in bb:
            augmented_bb = rotate90Deg_counter_clockwise(bounding_box, img_width)
            augmented_coordinates.append(augmented_bb)
        return augmented_coordinates, img_ccw_90


def rotate90Deg_counter_clockwise(bndbox, img_width):  # just passing width of image is enough for 90-degree rotation.
    x_min, y_min, x_max, y_max = bndbox
    new_xmin = y_min
    new_ymin = img_width - x_max
    new_xmax = y_max
    new_ymax = img_width - x_min
    return [new_xmin, new_ymin, new_xmax, new_ymax]


def rotate90Deg_clockwise(bndbox, img_width):  # just passing width of image is enough for 90-degree rotation.
    x_min, y_min, x_max, y_max = bndbox
    new_xmin = img_width - y_max  # Reflection about center X-line
    new_ymin = x_min
    new_xmax = img_width - y_min  # Reflection about center X-line
    new_ymax = x_max
    return [new_xmin, new_ymin, new_xmax, new_ymax]


def listToString(s):
    # initialize an empty string
    str1 = " "

    # return string
    return str1.join(str(s))


def voc_to_yolo(rotated_bb, rotated_co):
    height, width, _ = rotated_bb.shape
    actual_coordinates_list = []
    for i in range(len(rotated_co)):
        actual_co = pbx.convert_bbox(rotated_co[i], from_type="voc", to_type="yolo", image_size=(width, height))
        actual_coordinates_list.append(list(actual_co))
    return actual_coordinates_list


def save_file(jpg, txt, indexes):
    id_ = str(uuid.uuid1())
    jpg_file = to_save + id_ + '.jpg'
    txt_file = to_save + id_ + '.txt'
    cv2.imwrite(jpg_file, jpg)
    co_ordinates = voc_to_yolo(jpg, txt)
    f = open(txt_file, 'w')
    for i in range(len(indexes)):
        each_co = [str(each_coord) for each_coord in co_ordinates[i]]
        each_line = str(indexes[i]) + ' '
        for co in each_co[:-1]:
            each_line += co + ' '
        each_line += each_co[-1]
        f.write(each_line)
        f.write('\n')


def v_h_cw_cw(actual_co_ordinates, image, class_ind):
    save_file(image, actual_co_ordinates, class_ind)
    # vertical
    x = ImageAugmentation
    q_v, i_v = x.vertical_flip_image(actual_co_ordinates, image)
    save_file(i_v, q_v, class_ind)
    # horizontal
    q_h1, i_h1 = x.flip_horizontal(actual_co_ordinates, image)
    save_file(i_h1, q_h1, class_ind)
    q_h2, i_h2 = x.flip_horizontal(q_v, i_v)
    save_file(i_h2, q_h2, class_ind)

    # clockwise
    q_cw1, i_cw1 = x.rotate_90_deg_clockwise(actual_co_ordinates, image)
    save_file(i_cw1, q_cw1, class_ind)
    q_cw2, i_cw2 = x.rotate_90_deg_clockwise(q_h1, i_h1)
    save_file(i_cw2, q_cw2, class_ind)
    q_cw3, i_cw3 = x.rotate_90_deg_clockwise(q_h2, i_h2)
    save_file(i_cw3, q_cw3, class_ind)
    q_cw4, i_cw4 = x.rotate_90_deg_clockwise(q_v, i_v)
    save_file(i_cw4, q_cw4, class_ind)

    # anti-clock
    q_acw1, i_acw1 = x.rotate_90_deg_counter_clockwise(actual_co_ordinates, image)
    save_file(i_acw1, q_acw1, class_ind)
    q_acw2, i_acw2 = x.rotate_90_deg_counter_clockwise(q_h1, i_h1)
    save_file(i_acw2, q_acw2, class_ind)
    q_acw3, i_acw3 = x.rotate_90_deg_counter_clockwise(q_h2, i_h2)
    save_file(i_acw3, q_acw3, class_ind)
    q_acw4, i_acw4 = x.rotate_90_deg_counter_clockwise(q_v, i_v)
    save_file(i_acw4, q_acw4, class_ind)


def v_h(actual_co_ordinates, image, class_ind):
    save_file(image, actual_co_ordinates, class_ind)
    # vertical
    x = ImageAugmentation
    q_v, i_v = x.vertical_flip_image(actual_co_ordinates, image)
    save_file(i_v, q_v, class_ind)
    # horizontal
    q_h1, i_h1 = x.flip_horizontal(actual_co_ordinates, image)
    save_file(i_h1, q_h1, class_ind)
    q_h2, i_h2 = x.flip_horizontal(q_v, i_v)
    save_file(i_h2, q_h2, class_ind)


to_save = "E:/Kalypso_new/New_Dataset_Segrigated/temp/"
parent_dir = r"E:\Kalypso\Old_new_merge"
for filename in os.listdir(parent_dir):
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(parent_dir, filename))
        h, w, _ = img.shape

        file, ext = filename.split(".")
        txt_name = file + ".txt"

        with open(os.path.join(parent_dir, txt_name)) as f:
            lines = f.readlines()
            # print(lines)
            class_indexes = []
            actual_coordinates = []
            for line in lines:
                line = line[:len(line) - 1]
                line_li = line.split(" ")
                coord = line_li[1:]
                class_index = int(line_li[0])
                class_indexes.append(class_index)
                class_indexes_set = set(class_indexes)
                coord_int = [eval(i) for i in coord]
                coord_set = tuple(coord_int)
                actual_coord = pbx.convert_bbox(coord_set, from_type="yolo", to_type="voc", image_size=(w, h))
                actual_coordinates.append(list(actual_coord))

                x = ImageAugmentation
                # cv2.imshow("out", img)
                # cv2.waitKey(0)

            # dic = {'1': v_h, '3': v_h, '5': v_h}
            if 1 in class_indexes_set or 3 in class_indexes_set or 5 in class_indexes_set:
                v_h(actual_coordinates, img, class_indexes)
            else:
                v_h_cw_cw(actual_coordinates, img, class_indexes)
