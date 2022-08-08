import os 
import xml.etree.ElementTree as ET

def get_new_tree(fp):
    data_root = '/raid/data/parkingslot/img/'
    tree = ET.parse(fp)
    root = tree.getroot()
    fn = root.find('filename').text
    fn = fn.replace('JPG', 'jpg')
    root.find('filename').text = fn
    root.find('path').text = data_root + fn

    return tree

def get_tree(fp):
    tree = ET.parse(fp)
    return tree

def get_bbox(fp):
    tree = ET.parse(fp)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bnd_box = obj.find('bndbox')
        bbox = [
            int(float(bnd_box.find('xmin').text)),
            int(float(bnd_box.find('ymin').text)),
            int(float(bnd_box.find('xmax').text)),
            int(float(bnd_box.find('ymax').text))
        ]
        ob = {'label':label, 'bbox':bbox}
        objects.append(ob)
    return objects
    
def merge(tree1, tree2):
    root1 = tree1.getroot()
    root2 = tree2.getroot()
    # import ipdb
    # ipdb.set_trace()
    objs = root2.findall('object')
    for obj in objs:
        root1.append(obj)
    return tree1


CLASS = ['道闸','地面垃圾','地面破损及凹坑','地锁','减速带','路障','限位器','雪糕筒']
imgs = []
imgs_all = []
for cls in CLASS:
    img_path = 'srcimg/' + cls
    anno_path = 'xml/' + cls
    imgs = os.listdir(img_path) 
    annos = os.listdir(anno_path)
    print(len(imgs))
    print(len(annos))
    imgs.sort()
    annos.sort()
    for img in imgs:
        prefix = img.split('.')[0]
        if prefix + '.xml' in annos:    # have corrosponding annotation
            img_fp = prefix + '.jpg'
            anno_fp = prefix + '.xml'
            if not os.path.exists('img/'+img_fp):    # no repeat image
                os.system(f'cp srcimg/{cls}/{img} img/{img_fp}')
                tree = get_new_tree(f'xml/{cls}/{anno_fp}')
                tree.write(f'label/{anno_fp}')
            else: # merge annos
                assert os.path.getsize('img/'+img_fp) == os.path.getsize(f'srcimg/{cls}/{img}'), \
                    'same filename, different filesize in {cls}/{img_fp}'
                print(f'merge srcimg/{cls}/{img}')
                tree = get_tree(f'label/{anno_fp}')
                new_tree = get_tree(f'xml/{cls}/{anno_fp}')
                tree = merge(tree, new_tree)
                tree.write(f'label/{anno_fp}')
        else:
            print(f'no annotation in {cls}{img_fp}')

            
