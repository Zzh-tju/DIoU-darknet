
def intersection(a, b):
    '''
        input: 2 boxes (a,b)
        output: overlapping area, if any
    '''
    top = max(a[0], b[0])
    left = max(a[1], b[1])
    bottom = min(a[2], b[2])
    right = min(a[3], b[3])
    h = max(bottom - top, 0)
    w = max(right - left, 0)
    return h * w

def union(a, b):
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    return a_area + b_area - intersection(a,b)

def c(a, b):
    '''
        input: 2 boxes (a,b)
        output: smallest enclosing bounding box
    '''
    top = min(a[0], b[0])
    left = min(a[1], b[1])
    bottom = max(a[2], b[2])
    right = max(a[3], b[3])
    h = max(bottom - top, 0)
    w = max(right - left, 0)
    return h * w

def iou(a, b):
    '''
        input: 2 boxes (a,b)
        output: Itersection/Union
    '''
    U = union(a,b)
    if U == 0:
        return 0
    return intersection(a,b) / U


def giou(a, b):
    '''
        input: 2 boxes (a,b)
        output: Itersection/Union - (c - U)/c
    '''
    I = intersection(a,b)
    U = union(a,b)
    C = c(a,b)
    iou_term = (I / U) if U > 0 else 0
    giou_term = ((C - U) / C) if C > 0 else 0
    #print("  I: %f, U: %f, C: %f, iou_term: %f, giou_term: %f"%(I,U,C,iou_term,giou_term))
    return iou_term - giou_term

