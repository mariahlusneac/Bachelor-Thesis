from time import sleep
import pyautogui

sleep(5)
i = 0
no_iterations = 10
while i < no_iterations:
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save(f'C:\\Users\\Maria\\Desktop\\screenshot{i}.png')
    i += 1
    sleep(5)