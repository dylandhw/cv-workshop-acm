import cv2 
import matplotlib.pyplot as plt 


def static_edge_detection(file_name):
    # load our image
    image = cv2.imread("examples/city.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")
    plt.show()

    # apply a filter 
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    plt.imshow(blur, cmap="gray")
    plt.title("Blurred - Reduced Noise")
    plt.axis("off")
    plt.show()


    # edge detection
    edges = cv2.Canny(blur, 100, 200)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edge Detection")
    plt.axis("off")
    plt.show()


def live_webcam():
    capture = cv2.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        cv2.imshow("Live Edges", edges)
        
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    capture.release()
    cv2.destroyAllWindows()

def main():
    static_edge_detection("examples/city.jpg")
    live_webcam()   
if __name__ == "__main__":
    main()