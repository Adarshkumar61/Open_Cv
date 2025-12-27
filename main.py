print("Select an OpenCV module:")
print("1. Blue Color Detection")
print("2. Face Detection")
print("3. Face Edge Gray Detection")
print("4. Face Mask Detection")
print("5. HSV Value Detection")
print("6. Motion Detection")
print("7. Object Detection with Label")

choice = input("Enter choice (1-7) or 'Q' to quit: ")

if choice == "1":
    from modules import color_detection
elif choice == "2":
    from modules import face_detection
elif choice == "3":
    from modules import face_edge_gray_detection
elif choice == "4":
    from modules import face_mask_detection
elif choice == "5":
    from modules import HSV_value_detection
elif choice == "6":
    from modules import motion_detection
elif choice == "7":
    from modules import object_detection_with_label
elif choice == "q".lower():
    print("Exiting the program.")
    exit()
else:
    print("Invalid choice. Please select a valid option.")
