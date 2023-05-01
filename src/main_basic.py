from panorama_basic import panorama_basic
from perspective import postprocess

print('******************************************* Program Start *******************************************')
print('************************You are calling the basic panorama stitching program*************************')
print('There are 4 test videos choices: test_video_1, test_video_2, test_video_3, test_video_4')
test_name = input('Please type which video you want to input: ')
while test_name not in ['test_video_1', 'test_video_2', 'test_video_3', 'test_video_4']:
    test_name = input('Invalid input. Please type which video you want to input: ')
print(f'You have chosen {test_name}.')
print('')     

h, w = [1080, 1920]
offset_x = int(w*3)
offset_y = int(h*1.5)
print(f'This size of this video frame is {w}x{h}.')
print(f'The default panorama size is 3 times w and 1.5 time.')
x_time = input('Please type how many times you want the width (w) of panorama be: ')
y_time = input('Please type how many times you want the height (h) of panorama be: ')
offset_x = int(w*float(x_time))
offset_y = int(h*float(y_time))
print(f'The raw panorama size will be: {offset_x}x{offset_y}.')
print('') 

### Flag to determine whether whether write frame-by-frame results when stitching
if_write_frame = False
if_write_frame = input('Please enter whether you want to write frame-by-frame results [Ture, False]: ')
while if_write_frame not in ['True', 'False']:
    if_write_frame = input('Invalid input. Please enter whether you want to write frame-by-frame results [Ture, False]: ')
if_write_frame = bool(if_write_frame)
print('') 

panorama_basic(test_name, offset_x, offset_y, if_write_frame)
print('')

postprocess(test_name)
print('******************************************** Program End ********************************************')