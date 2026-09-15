import os
import time
import pyautogui
from pynput import mouse #滑鼠控制套件
import cv2   #視窗套件
import mss   #截圖套件
import numpy as np 
from ultralytics import YOLO  
import threading
import queue



print("請進入遊戲！")
def on_click0():
        print('已進入遊戲！請開啟撈魚畫面')
        return False
with mouse.Listener(on_click=on_click0) as listener:
    listener.join() 


coords = []
x1, y1, x2, y2 = 0, 0, 0, 0

def on_click1(x, y, button, pressed):
    if pressed:  
        coords.append((int(x), int(y)))
        global x1, y1, x2, y2
        print(f"已記錄第 {len(coords)} 個座標: ({int(x)}, {int(y)})")
        if len(coords) == 3:
             print("請點擊水桶位置")#coords[3]是水桶位置
        if len(coords) == 4:
            print("請點擊魚池左上角位置")#coords[4]是魚池左上角位置
        if len(coords) == 5:
            x1 = int(x)
            y1 = int(y)
            print("請點擊魚池右下角位置")#coords[5]是魚池右下角位置
        if len(coords) == 6:
            x2 = int(x)
            y2 = int(y)
            return False
with mouse.Listener(on_click=on_click1) as listener:
    listener.join()  


    
print("位置記錄完成！")
time.sleep(0.5)
print("請返回遊戲主畫面，即將開始自動撈魚程式")
time.sleep(2)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("開始撈魚！")





is_fishing = True  
action_queue = queue.Queue(maxsize=1)  
is_ready = True  # 用於控制自動撈魚的開關


def AutoControl():
    while is_fishing:
        try:
            target_coord = action_queue.get(timeout=0.1)
            pyautogui.moveTo(target_coord[0], target_coord[1], duration=0.15)
            pyautogui.click(target_coord)
            time.sleep(0.1)  # 點擊後稍作延遲，避免過快
            pyautogui.moveTo(coords[3][0], coords[3][1], duration=0.15)
            pyautogui.click(coords[3])  # 點擊水桶位置
            global last_collect_time
            last_collect_time = time.time()  # 更新最後收集時間
            global is_ready 
            is_ready = True  # 設定為 True，表示可以進行下一次收集
        except queue.Empty:
            continue

threading.Thread(target=AutoControl, daemon=True).start()




fishingtime = 5
COLLECT_INTERVAL = 0.3
last_collect_time = time.time()+COLLECT_INTERVAL
now = time.time()

model = YOLO('best.pt')
CONFIDENCE_THRSHOLD = 0.6  # 信心門檻 (可依畫面微調，如 0.35 ~ 0.5)

first_time = True  # 用於控制第一次收集的標誌
successtimes = 0

for i in range(fishingtime):
    start = False
    print(f"第 {i+1} 次撈魚開始！")

    pyautogui.moveTo(coords[0][0], coords[0][1], duration=0.15)
    pyautogui.click(coords[0])
    time.sleep(0.5)
    pyautogui.moveTo(coords[1][0], coords[1][1], duration=0.15)
    pyautogui.click(coords[1])
    time.sleep(0.5)
    pyautogui.moveTo(coords[2][0], coords[2][1], duration=0.15) 
    pyautogui.click(coords[2])
    time.sleep(0.5)

    screen_w, screen_h = pyautogui.size()
    with mss.MSS() as sct:
        scale_x = sct.monitors[1]["width"] / screen_w
        scale_y = sct.monitors[1]["height"] / screen_h

        GAME_LEFT = x1
        GAME_TOP = y1
        GAME_W = x2 - x1
        GAME_H = y2 -y1


        window_name = "FISH TRACKER"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, GAME_W, GAME_H)
        cv2.moveWindow(window_name,int (screen_w/2),int (screen_h/2))

        # 轉為截圖像素座標
        capture_region = {
            "left": int(GAME_LEFT * scale_x),
            "top": int(GAME_TOP * scale_y),
            "width": int(GAME_W * scale_x),
            "height": int(GAME_H * scale_y)
        }


        prev_time = time.time()

        with mss.MSS() as sct:
            monitor = sct.monitors[1]

            try:
                
                while True:
                    # 1. 極速截取畫面
                    screenshot = sct.grab(capture_region)
                    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
                    cv2.imshow("FISH TRACKER", frame)
                    

                    if first_time:
                        first_time = False
                        #print("快錄影！然後把滑鼠移到遊戲頁面")
                        cv2.waitKey(1000)  # 等待 1 秒，給視窗一些時間來初始化
                        #time.sleep(5)  
                        pyautogui.click()

                    # 2. YOLO 模型推論
                    results = model.predict(source=frame, conf=CONFIDENCE_THRSHOLD, verbose=False)

                    fish_count = 0

                    # 3. 繪製每隻魚的方框與標籤
                    for r in results:

                        now = time.time()

                        if (now - last_collect_time >= COLLECT_INTERVAL) and is_ready and len(r.boxes) > 3 :
                            # 1. 把所有魚的 (x, y) 拿出來轉成普通的 Python list
                            fish_list = [box.xywh[0][:2].tolist() for box in r.boxes]

                            # 2. 定義一個小函式：數數看某隻魚身邊半徑 150 像素內有幾個鄰居
                            def count_friends(target):
                                tx, ty = target
                                
                                return sum(1 for (x, y) in fish_list if (x - tx)**2 + (y - ty)**2 <= 22500)

                            # 3. 挑出身邊朋友最多的那隻魚！
                            best_fish = max(fish_list, key=count_friends)
                            group_cx, group_cy = best_fish
                            cluster = [
                                (x, y) for (x, y) in fish_list if (x - group_cx) ** 2 + (y - group_cy) ** 2 <= 20000
                            ]

                            # 3. 計算這個小圈子的中心
                            group_cx = sum(x for (x, y) in cluster) / len(cluster)
                            group_cy = sum(y for (x, y) in cluster) / len(cluster)
                            group_cx = max(min(group_cx, x2 - 80), x1 + 80)  # 限制在遊戲視窗內
                            group_cy = max(min(group_cy, y2 - 80), y1 + 80)  # 限制在遊戲視窗內

                            if len(cluster) >= 3:
                                action_queue.put_nowait((int(group_cx), int(group_cy)))  # 放入佇列，讓 AutoControl 處理點擊
                                is_ready = False  # 設定為 False，直到下一次收集時間到來
                             

                                
                        for box in r.boxes:
                            fish_count += 1
                            x3, y3, x4, y4 = map(int, box.xyxy[0].tolist())

                            conf = float(box.conf[0])

                            # 科技白/青色方框 (BGR: 255, 255, 255 純白風格更像原圖監視器)
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)

                            # 方框上方標註文字
                            label = f"FISH {conf*100:.0f}%"
                            cv2.putText(
                                frame, 
                                label, 
                                (x3, max(y3 - 8, 18)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 255, 255), 
                                1, 
                                cv2.LINE_AA
                            )
                        
                    # 4. 計算 FPS 與儀表顯示
                    curr_time = time.time()
                    fps = 1.0 / (curr_time - prev_time + 1e-6)
                    prev_time = curr_time

                    cv2.putText(frame, f"FPS: {fps:.1f}", (25, 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"DETECTED: {fish_count}", (25, 75), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                    # 5. 輸出顯示畫面
                    cv2.imshow("FISH TRACKER", frame)



                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break #跳出while迴圈

                    if now - last_collect_time > 3 and len(r.boxes) < 3:
                        try:
                            button_location = pyautogui.locateOnScreen("endbutton.png",confidence=0.9)
                            # 找到了才會執行以下邏輯
                            button_center = pyautogui.center(button_location)
                            real_x = button_center.x / 2
                            real_y = button_center.y / 2
                            pyautogui.moveTo(real_x, real_y, duration=0.2)
                            pyautogui.click()
                            print("已成功點擊結束按鍵！")
                            successtimes += 1
                            last_collect_time = time.time() + 5  # 換場時間
                        except (pyautogui.ImageNotFoundException, Exception):
                            print("找不到按鍵，請檢查按鍵是否被遮擋，或是截圖解析度不對。")
                            last_collect_time = time.time() + 5  # 換場時間
                        break #跳出while迴圈

            except KeyboardInterrupt:
                pass


print(f"成功撈魚次數: {successtimes} 次")
print(">>> 準備銷毀所有視窗...")            
is_fishing = False #關掉撈魚def
cv2.destroyAllWindows()
cv2.waitKey(1000)  # 確保所有 OpenCV 視窗都被銷毀

print(">>> 視窗銷毀完成！")


    
