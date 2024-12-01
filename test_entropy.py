from inr_utils.entropy import relative_entropy_rgb, parrot

entropy_parrot = relative_entropy_rgb(parrot)
print(entropy_parrot)