
import os
def main():
    print('hi')
    images_root = 'data/hundred_pounds_jpg/'
    img_names = os.listdir(images_root)
    with open("example1.txt", "w", encoding='utf-8') as f:
        for img in img_names:
            f.write(img)
            f.write('#0')
            f.write('\t')
            f.write(' 100 جنية مصري')
            f.write('\n')
            f.write(img)
            f.write('#1')
            f.write('\t')
            f.write(' 100 جنية')
            f.write('\n')
            f.write(img)
            f.write('#2')
            f.write('\t')
            f.write('عملة نقدية فئة 100 جنية')
            f.write('\n')

    with open("img.txt", "w") as f1:
        for img in img_names:
            f1.write(img)
            f1.write('\n')
    captions_file_text = load_file_text('example1.txt')
    captions = get_captions(captions_file_text)

def load_file_text(file_path):
        """reads and returns text in captions file"""
        file = open(file_path, 'r', encoding='utf-8')
        all_text = file.read()
        file.close()
        return all_text

def get_captions(file_text):

        cpts = {}
        # loop through lines
        for line in file_text.split('\n'):  # each line contains image name & its caption separated by tab
            # split by tabs
            img_cpt = line.split('\t')
            if len(img_cpt) < 2: continue
            img, cpt = img_cpt
            # remove image extension & index (remove everything befor the dot)
            img_name = img.split('.')[0]
            # add to dictionary
            if img_name not in cpts:
                cpts[img_name] = [cpt]
            else:
                cpts[img_name].append(cpt)
        return cpts
if __name__ == '__main__':
    main()
