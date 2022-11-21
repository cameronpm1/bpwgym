import math

def calcIntersection(x1,y1,x2,y2,r1,r2):
    if(math.sqrt((x1-x2)**2+(y1-y2)**2) > r1 + r2):
        return None
    elif y1==y2:
        x = ((r1**2-r2**2)-(x1**2-x2**2))/(-2*(x1-x2))
        y_diff = math.sqrt(r1**2 - (x-x1)**2)
        y_1 = y1 - y_diff
        y_2 = y1 + y_diff
        return(x,y_1,x,y_2)
    elif x1==x2:
        y = ((r1**2-r2**2)-(y1**2-y2**2))/(-2*(y1-y2))
        x_diff = math.sqrt(r1**2 - (y-y1)**2)
        x_1 = x1 - x_diff
        x_2 = x1 + x_diff
        return(x_1,y,x_2,y)
    else:
        C2 = -1*(r1**2-r2**2)/(2*(y1-y2)) + (x1**2-x2**2)/(2*(y1-y2)) + (y1**2-y2**2)/(2*(y1-y2))
        C1 = -1*(x1-x2)/(y1-y2)
        a = C1**2+1
        b = 2*C1*C2-2*C1*y1-2*x1
        c = -2*C2*y1+y1**2+C2**2+x1**2-r1**2
        desc = b**2-4*a*c
        if desc < 0:
            return None
        else:
            x_1 = (-1*b + math.sqrt(desc))/(2*a)
            x_2 = (-1*b - math.sqrt(desc))/(2*a)
            y_1 = C1*x_1+C2
            y_2 = C1*x_2+C2
        return (x_1,y_1,x_2,y_2)

def isIntersection(x1,y1,x2,y2,r1,r2):
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    if dist <= r1+r2:
        return True
    else:
        return False

def findRangePointOfIntersection(x1,y1,r,len1,len2,angle_range,tol):
    #x1,y1 - position of end point of first link
    #r - length of link to solve for
    #len1,len2 - lengths of free links
    #angle_range - angle range to test
    #                               *
    #                               | -> controlled link (x1,x2)
    #                               |
    #                      *        *
    #  link to solve for <- \      /
    #                        \*--*/ <- free link 2
    #                          ^
    #                          |
    #                         free link 1
    const = (len1+len2)**2-(x1**2+y1**2+r**2)/(-2*r)
    done = False
    a1 = angle_range[0]
    a2 = angle_range[1]
    RHS1 = x2 * math.cos(math.radians(a1)) + y2 * math.sin(math.radians(a1))
    RHS2 = x2 * math.cos(math.radians(a2)) + y2 * math.sin(math.radians(a2))
    if RHS1 > const and RHS2 > const:
        return None
    else:
        while not done:
            if abs(RHS1-const) < abs(RHS2-const):
                a2 = a1 + (a2-a1) / 2
                RHS2 = x2 * math.cos(math.radians(a2)) + y2 * math.sin(math.radians(a2))
            else:
                a1 = a1 + (a2 - a1) / 2
                RHS1 = x2 * math.cos(math.radians(a1)) + y2 * math.sin(math.radians(a1))
            if abs(RHS1 - const) < tol:
                done = True
                angle = a1
            elif abs(RHS1-const) < tol:
                done = True
                angle = a2
    return angle