import torch


def findRegionCenter(x, y, n,xmin1,xmin2,xmax1,xmax2):
    minX, maxX = xmin1,xmax1
    minY, maxY = xmin2,xmax2

    # Calculate the width and height of each region
    delta_x = (maxX - minX) / n
    delta_y = (maxY - minY) / n

    # Determine the region index for the input coordinates
    region_x = torch.clamp(torch.floor((x - minX) / delta_x) + 1, 1, n).long()
    region_y = torch.clamp(torch.floor((y - minY) / delta_y) + 1, 1, n).long()

    # Calculate the coordinates of the center point in the identified region
    center_point_x = minX + (region_x - 0.5) * delta_x
    center_point_y = minY + (region_y - 0.5) * delta_y

    return center_point_x, center_point_y, region_x, region_y

def renewl3(x, y, v2, v3, n, l,xmin1,xmin2,xmax1,xmax2):
    if x > xmax1 or x < xmin1 or y > xmax2 or y < xmin2:
        out2 = v2
        out3 = v3
    else:
        cx, cy, indx, indy = findRegionCenter(x, y, n,xmin1,xmin2,xmax1,xmax2)
        indx,indy=indx-1,indy-1
        distance = torch.norm(torch.tensor([x - cx, y - cy]))
        w = torch.exp(-1.414 * distance * n / 4)
        l[1, indx, indy] = (l[1, indx, indy] * l[0, indx, indy] + v2 * w) / (l[0, indx, indy] + w)
        l[2, indx, indy] = (l[2, indx, indy] * l[0, indx, indy] + v3 * w) / (l[0, indx, indy] + w)
        l[0, indx, indy] = l[0, indx, indy] + w
        out2 = l[1, indx, indy]
        out3 = l[2, indx, indy]

    return out2, out3, l

def ptol(o, n,xmin1,xmin2,xmax1,xmax2):
    minX, maxX = xmin1,xmax1
    minY, maxY = xmin2,xmax2
    sample = torch.zeros((1500, 5), dtype=torch.float32)

    # Calculate the width and height of each region
    deltaX = (maxX - minX) / n
    deltaY = (maxY - minY) / n

    index = 0
    for i in range(0, n ):
        for j in range(0, n ):
            if o[0, i , j ] != 0:
                cx = minX + (i + 0.5) * deltaX
                cy = minY + (j + 0.5) * deltaY
                sample[index, :] = torch.tensor([cx, cy, o[0, i, j], o[1, i, j], o[2, i, j]], dtype=torch.float32)
                index += 1

    return sample

def findtheta(fitness, k, x, N):
    fitness=fitness.squeeze()
    _, index = torch.sort(fitness)
    weak = index[N-1]
    index = index[:k]
    elite_value = fitness[index]
    elite_x = x[index, :]
    fmax = torch.max(elite_value)
    fmin = torch.min(elite_value)
    m = torch.exp((fmax - elite_value) / (fmax - fmin))
    W = m / torch.sum(m)
    theta = torch.matmul(W, elite_x)

    return theta, weak

def corrxv1(input, xvmin, xvmax):
    input_shape = input.size()
    output = torch.ones(input_shape) * (input > xvmax) * xvmax + torch.ones(input_shape) * (input < xvmin) * xvmin + input * ((input <= xvmax) & (input >= xvmin))
    return output
