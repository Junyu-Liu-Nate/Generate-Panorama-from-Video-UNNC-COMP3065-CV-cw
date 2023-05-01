from panorama import panorama
from perspective import postprocess

### Choices for test_name: ['test_video_1', 'test_video_2', 'test_video_3'] ### 
test_name = 'test_video_1'

h, w = [1080, 1920]
offset_x = int(w*3)
offset_y = int(h*1.5)

panorama(test_name, offset_x, offset_y)
postprocess(test_name)