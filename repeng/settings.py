VERBOSE = False

LOW_MEMORY = False  # When extracting the representations for a given activation, 
# if LOW_MEMORY is True then we stack arrays as they arrive. If it's False, we stack them at the end.
# This shoudldn't matter in most cases, except maybe if you have many many examples.
