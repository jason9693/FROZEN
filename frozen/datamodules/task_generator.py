BUILTIN_INSTRUCTION_SET = dict(
    image_captioning=[
        'What does the image describe?',
        'What do you think about the image?',
        'What do you think about the picture?',
        'Describe the image.',
        'What does the picture describe?',
        'What is the situation with the image?',
        'What does the image depict?',
        'Explain the image.',
        'What is described in the image?',
        'What is happening to the image?'
    ],
    text2image_detection=[
        'Which region does the text "<text>" describe?',
        'Which region does the text "<text>" depict?',
        'Where is this text "<text>" describing?',
        'Where is this text "<text>" depicting?',
        'Where is the location that this text "<text>" describes?',
        'Where is the location that this text "<text>" depicts?',
        'Where is the region that this text "<text>" describes?',
        'Where is the region that this text "<text>" depicts?',
        'Find the location that this text "<text>" describes.',
        'Find the location that this text "<text>" depicts.',
        'Find the region that this text "<text>" describes.',
        'Find the region that this text "<text>" depicts.',
        'Select the location that this text "<text>" describes.',
        'Select the location that this text "<text>" depicts.',
        'Select the region that this text "<text>" describes.',
        'Select the region that this text "<text>" depicts.'
    ],
    classification=[
        'What does the image describe?',
        'What does the image depict?',
        'What is this?',
        'Explain the image in a word.'
    ],
    text2image_generation=[
        'What is the complete image? caption: <text>',
        'What is the proper image for this text? caption: <text>',
        'Generate an image from this text. caption: <text>',
        'Draw an image for this text. caption: <text>'
    ],
    visual_entailment=[
        'Can image and text "<text>" imply text "<text>"?',
        'Can image and text "<text>" entail text "<text>"?'
    ],
    style=[
        'What is the <style> style image of this image?'
    ]
)




