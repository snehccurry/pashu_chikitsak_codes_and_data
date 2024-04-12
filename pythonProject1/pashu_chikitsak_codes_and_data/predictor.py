from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO('D:\\TEST\\final_bca_project\\finalyearproject\\test\\runs\\classify\\train\\weights\\best.pt')

# Perform object detection on the image
#results = model("https://static.candadiancattlemen.ca/wp-content/uploads/2014/09/Footrot-2-e1412702218599.jpg")

#results=model("https://www.alltech.com/sites/default/files/styles/16_9_large/public/2019-09/cattle%20%20footrot%20BLOG.png.jpg")

#results=model("https://c8.alamy.com/comp/BTGYNE/legs-feet-and-hooves-of-friesian-heifer-young-cow-BTGYNE.jpg")

image_list=[
            "https://www.alltech.com/sites/default/files/styles/16_9_large/public/2019-09/cattle%20%20footrot%20BLOG.png.jpg",
            "https://c8.alamy.com/comp/BTGYNE/legs-feet-and-hooves-of-friesian-heifer-young-cow-BTGYNE.jpg",
            "D:\\TEST\\final_bca_project\\finalyearproject\\test\\animal_dataset\\val\\diseased\\diseased (1).jpg",
            "D:\\TEST\\final_bca_project\\finalyearproject\\test\\animal_dataset\\val\\diseased\\diseased (2).jpg",
            "D:\\TEST\\final_bca_project\\finalyearproject\\test\\animal_dataset\\val\\diseased\\diseased (3).jpg",
            "D:\\TEST\\final_bca_project\\finalyearproject\\test\\animal_dataset\\val\\diseased\\diseased (4).jpg",
            "D:\\TEST\\final_bca_project\\finalyearproject\\test\\animal_dataset\\val\\diseased\\diseased (5).jpg",
            "https://c8.alamy.com/comp/BTGYNE/legs-feet-and-hooves-of-friesian-heifer-young-cow-BTGYNE.jpg",
            "https://www.alltech.com/sites/default/files/styles/16_9_large/public/2019-09/cattle%20%20footrot%20BLOG.png.jpg",
            #"https://static.candadiancattlemen.ca/wp-content/uploads/2014/09/Footrot-2-e1412702218599.jpg"
            ]


#image_list=['https://agritech.tnau.ac.in/animal_husbandry/images/Cow_1.jpg']
#add image_list=['https://www.msdvetmanual.com/-/media/manual/veterinary/images/s/e/v/severe-axial-fissure-cow-cramer-sized.jpg']
#image_list=['https://static.grainews.ca/wp-content/uploads/2017/05/Digital-Dermatitis-credit-file-CanCattlemen.jpg']

#add #image_list=['https://image.slidesharecdn.com/11diseasesofmusculoskeletalsystempart2-200615194640/85/11-diseases-of-musculoskeletal-system-part-2-8-320.jpg']

#image_list=['https://www.biopharmachemie.com/uploads/images/news/ky-thuat/Gia%20S%E2%94%9C%E2%95%91c/h2%20(2).jpg']

for image in image_list:
    results = model(image)
    #results = model("D:\\TEST\\final_bca_project\\finalyearproject\\test\\animal_dataset\\val\\normal\\image (1).jpg")

    # Extract class names and probabilities
    names_dict = results[0].names

    probs = results[0].probs.tolist()
    print("Class Names:", names_dict)
    print("Probabilities:", probs)

    print(f"for image {image} \t\t : {names_dict[np.argmax(probs)]}")




#========================







