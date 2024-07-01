import seaborn as sns

color_palette = sns.color_palette("colorblind", 200)
color_assignment = {}


def get_color(familiy_name: str) -> str:
    if familiy_name in color_assignment.keys():
        return color_assignment[familiy_name]

    new_color = color_palette[len(color_assignment)]
    color_assignment[familiy_name] = '#{:02x}{:02x}{:02x}'.format(int(new_color[0] * 255), int(new_color[1] * 255), int(new_color[2] * 255))

    return color_assignment[familiy_name]


def _square_to_lower_triangle(dist_square):
    n = len(dist_square)
    dist_lower_triangle = []
    for i in range(n):
        row = []
        for j in range(i + 1):
            row.append(dist_square[i][j])
        dist_lower_triangle.append(row)
    return dist_lower_triangle
