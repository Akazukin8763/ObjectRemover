from object_remover import ObjectRemover


def main():
    # filepath = './resources/IMG_1722.jpg'
    filepath = './resources/640x360_Zebra.mp4'
    # filepath = './resources/640x360_Birds.mp4.mp4'
    # filepath = './resources/640x360_Eagle.mp4'
    # filepath = './resources/640x360_Giraffe.mp4'

    remover = ObjectRemover()
    remover.load(filepath)
    remover.run(learning_base=False)


if __name__ == '__main__':
    main()
