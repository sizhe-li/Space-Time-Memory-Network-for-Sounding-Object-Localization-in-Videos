import xml.etree.ElementTree as ET

def read_annotations_xml(annot_file):
    tree = ET.parse(annot_file)
    root = tree.getroot()

    orig_size = root.find("meta").find("task").find("original_size")
    H, W = orig_size.find("height").text, orig_size.find("width").text
    orig_size = (int(H), int(W))
    labels = {}

    for track in root.iter("track"):
        for box in track.iter("box"):
            if not int(box.get("outside")):
                frame_index = int(box.get("frame"))
                xtl, ytl, xbr, ybr = (
                    box.get("xtl"),
                    box.get("ytl"),
                    box.get("xbr"),
                    box.get("ybr"),
                )
                coord = (float(xtl), float(ytl), float(xbr), float(ybr))
                labels[frame_index] = labels.get(frame_index, []) + [coord]

    return orig_size, labels
