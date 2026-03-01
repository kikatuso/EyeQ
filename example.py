
from src.EyeQ import run_grading

if __name__ == '__main__':

    img_path = '/well/papiez/users/zwk579/Analysis/EyeQ/example_images/'
    run_grading(img_path,img_extension='.png',batch_size=1,verbose=True)
    