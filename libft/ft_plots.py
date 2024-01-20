import random


def ft_generate_random_hexa_color_codes(number_of_colors_to_generate):
    color_codes = []
    for i in range(number_of_colors_to_generate):
        color_codes.append("#" + ("%06x" % random.randint(0, 0xFFFFFF)))
    return color_codes
