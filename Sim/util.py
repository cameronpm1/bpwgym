import math

def calcIntersection(x1,y1,x2,y2,r1,r2):
    """
    solve for the intersections between two circles:
    x1,y1,r1 -> center point and radius of circle #1
    x2,y2,r2 -> center point and radius of circle #2
    returns two points of intersection if there is a solution
    returns None if there is no solution
    """
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

#<position name="joint013"   gear="1"  joint="joint013"/>
#<position name="joint113"   gear="1"  joint="joint113"/>
#<position name="joint014"   gear="1"  joint="joint014"/>
#<position name="joint114"   gear="1"  joint="joint114"/>
#<position name="joint022"   gear="1"  joint="joint022"/>
#<position name="joint122"   gear="1"  joint="joint122"/>
#<position name="joint032"   gear="1"  joint="joint032"/>
#<position name="joint132"   gear="1"  joint="joint132"/>
