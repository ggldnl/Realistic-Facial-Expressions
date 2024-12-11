import functools


def compose(*functions):
    """
    Compose multiple functions into a single callable.
    """
    return lambda x: functools.reduce(lambda acc, f: f(acc), functions, x)

def replace_underscores(s):
    return s.replace('_', ' ')

def remove_numbers(s):
    return ''.join(c for c in s if not c.isdigit())

def remove_extension(s):
    return s.split('.')[0]

def strip(s):
    return s.strip()

def wrap(s):
    return f"A human face with {s} expression"

# Predefined text generations
DEFAULT_TEXT_GENERATION = compose(replace_underscores, remove_numbers, remove_extension, strip, wrap)


if __name__ == '__main__':

    # Example usage for "1_neutral.ply"
    filename = "1_neutral.ply"
    process_filename = compose(replace_underscores, remove_numbers, remove_extension)
    print(f'{filename} -> replace_uderscores, remove_numbers, remove_extension -> {process_filename(filename)}')  # Output: "neutral"
    print(f'{filename} -> DEFAULT_TEXT_GENERATION -> {DEFAULT_TEXT_GENERATION(filename)}')  # Output: "A human face with a neutral expression"