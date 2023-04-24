# MTech_CBS

This is where steganography comes into play. Steganography is the art of hiding data behind other ordinary and often publicly available “host data”. In this work, we explore text-hiding behind images.
For an added layer of security, we explore the options of an alternative text.Here, we selected the dancing men text to encode our hidden text. We first generate a dancing men image of the
secret text which is embedded into the Least Significant Bits (LSB) of the Host Image. On the output end, we extract the embedded watermark. This watermark is then processed and
passed on through a neural network for character recognition and is then collated together to assemble the deciphered text. On measuring the Similarity scores between the reconstructed
text and the original text using the Cosine and the Lehvenstein distance metrics, we were able to obtain consistently high results. We also found that the processing time required for
reconstruction is a function of the text rather than the host image. To check the effect of adding the text on the host image, we used the metrics SSIM and PSNR and obtained SSIM scores very
close to 1 in all cases. This proved that text can be hidden in the host images without causing significant changes to the host image.
