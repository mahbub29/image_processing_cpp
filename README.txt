This folder contains C++ code for a number of image processing functions.
To run the functions, using the terminal, navigate into where the folder
"im1", i.e.

> cd [PATH]/ImageProcessing/im1

Once in "im1", using the following command run any one of the chosen
image processing functions:

> ./impro [FUNCTION] [IMAGE PATH] [ARGUMENTS]

where:

[FUNCTION]      choose out of the following functions (functions may also
                have compulsory/optional [ARGUMENTS):

                optimize   - will provide a selection menu of optimization
                             options in terminal
                dcorner    - will conduct corner detection
                dedge      - will conduct edge detection


[IMAGE PATH]    enter the full path of the image to conduct image
                processes on


[ARGUMENTS]     the arguments for each of the above named functions are:

                for "optimize":
                    default constraint setting is set to use color image.
                    Optionally, to use grayscale image enter "0" for
                    constraint ("1" will simply use color image).


                for "dcorner":
                    dcorner takes 4 optional arguments in the order,
                    
                    [WINDOW RADIUS] [OVERLAY ON/OFF] [THRESHOLD] [LIMIT]

                    [WINDOW RADIUS]  - Radius of corner detection window.
                                       Must be positive integer.
                                       DEFAULT = 1

                    [OVERLAY ON/OFF] - Set to 1 for RED overlay on image,
                                       identifying corners.
                                       Set to 0 for separate BLACK output
                                       image with corners marked in WHITE.
                                       DEFAULT = 1

                    [THRESHOLD]      - Cornerness threshold. Must be
                                       positive integer.
                                       DEFAULT = 3000

                    [LIMIT]          - Set the absolute value limit at
                                       which neighboring pixel intensities
                                       can vary from the analysed pixel.
                                       DEFAULT = 20

                    *NOTE*: The above 4 arguments are optional, however
                            to enter a latter optional argument the former
                            arguments must also be entered. For example to
                            set the [THRESHOLD] argument to 1500, please
                            enter:

                            ./impro dcorner 1 1 1500

                            here the [WINDOW RADIUS] has been set to 1,
                            [OVERLAY ON/OFF] has been set to 1 (TRUE) and
                            [THRESHOLD] has been set to 1500. [LIMIT] will
                            remain at the default (20).

                    Please note that this function uses a Harris corner
                    detection algorithm. The code also contains a Movarec
                    corner detection algorithm; should users want to try use
                    this the relevant lines can be changed and compiled in
                    the code.


                for "dedge":
                    dedge takes 2 optional arguments in the order

                    [OVERLAY ON/OFF] [THRESHOLD]

                    [OVERLAY ON/OFF] - Set to 1 for RED overlay on image,
                                       identifying edges.
                                       Set to 0 for separate BLACK output
                                       image with edges marked in WHITE.
                                       DEFAULT = 1

                    [THRESHOLD]      - This algorithm uses a Sobel image,
                                       where the Sobel kernel highlights
                                       edges and darkens non-edge pixels.
                                       Optionally set this value to set the
                                       pixel intensity threshold for pixels
                                       to be identified as edges.
                                       DEFAULT = 50

                    *NOTE*: The above 2 arguments are optional, however to
                            enter the 2nd argument the first must also be
                            filled.




