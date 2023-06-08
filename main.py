from ultralytics import YOLO
import telebot
import numpy as np
import cv2
import os

tmp = 0
bot = telebot.TeleBot("6251722523:AAEI7a-GW4dRneTM8LSnRgr-1swvpaAOAF0")
exec_path = os.getcwd()
model_path = os.path.join(exec_path, "for_cpu.pt")
model = YOLO(model_path)



@bot.message_handler(content_types=['text'])
def handle_message(message):

    mess = f"Salom  rasmni jo'nating"
    bot.send_message(message.chat.id, mess)


@bot.message_handler(content_types=['document'])
def handle_image(message):
    fileID = message.document.file_id
    file_info = bot.get_file(fileID)

    # yuqoridagi kod faylni yuklab olshga tayyorlaydi, quyidagi esa binary shaklida yuklab oladi
    downloaded_file = bot.download_file(file_info.file_path)

    # binarydan modelimiz o'qiy oladigan numpy arrayga o'tkazib olamiz
    img = cv2.imdecode(np.frombuffer(downloaded_file, np.uint8), -1)
    results = model(img)
    img = results[0].plot(probs=False,labels=False)

    # rasmni yana qayta jo'natish uchun binaryga o'tkazamiz
    success,img = cv2.imencode('.png', img)

    annotated_frame = img.tobytes()

    # foydalanuvchiga yuzlar belgilangan rasmni yuboramiz
    bot.send_photo(message.chat.id, annotated_frame)


bot.polling(none_stop=True)

