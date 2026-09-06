import time
import cv2
import mss
import numpy as np
from ultralytics import YOLO
import pyautogui



print("請在 3 秒內把滑鼠移到遊戲魚池的【左上角】...")
time.sleep(3)
x1, y1 = pyautogui.position()
print(f"左上角座標: ({x1}, {y1})")

print("請在 3 秒內把滑鼠移到遊戲魚池的【右下角】...")
time.sleep(3)
x2, y2 = pyautogui.position()
print(f"右下角座標: ({x2}, {y2})")

print(f"\n你的魚池區域設定：")
print(f"REGION = {{'left': {x1}, 'top': {y1}, 'width': {x2 - x1}, 'height': {y2 - y1}}}")


screen_w, screen_h = pyautogui.size()
with mss.mss() as sct:
    scale_x = sct.monitors[1]["width"] / screen_w
    scale_y = sct.monitors[1]["height"] / screen_h

GAME_LEFT = x1
GAME_TOP = y1
GAME_W = x2 - x1
GAME_H = y2 -y1

window_name = "FISH TRACKER"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 360)
cv2.moveWindow(window_name, screen_w - 660, screen_h - 420)

# 轉為截圖像素座標
capture_region = {
    "left": int(GAME_LEFT * scale_x),
    "top": int(GAME_TOP * scale_y),
    "width": int(GAME_W * scale_x),
    "height": int(GAME_H * scale_y)
}


# 1. 載入模型
model = YOLO('best.pt')


# 2. 參數設定
CONFIDENCE_THRESHOLD = 0.5  # 信心門檻 (可依畫面微調，如 0.35 ~ 0.5)

print("=== 魚群純視覺追蹤視窗啟動 ===")
print("請將視窗移至一旁觀賞。按鍵盤上的 'q' 鍵可退出。\n")

cv2.namedWindow("FISH TRACKER", cv2.WINDOW_NORMAL)
cv2.resizeWindow("FISH TRACKER", 1024, 576)

prev_time = time.time()

with mss.mss() as sct:
    monitor = sct.monitors[1]

    try:
        while True:
            # 1. 極速截取畫面
            screenshot = sct.grab(capture_region)
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)

            # 2. YOLO 模型推論
            results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            fish_count = 0

            # 3. 繪製每隻魚的方框與標籤
            for r in results:
                for box in r.boxes:
                    fish_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])

                    # 科技白/青色方框 (BGR: 255, 255, 255 純白風格更像原圖監視器)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                    # 方框上方標註文字
                    label = f"FISH {conf*100:.0f}%"
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, max(y1 - 8, 18)), 
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
                break

    except KeyboardInterrupt:
        pass

cv2.destroyAllWindows()
print("視窗已順利關閉。")
