# 271101_workshop_assignment
หลักการทำงานของโปรแกรม 
    การใช้งานด้านล่างนี้ทำงานโดยการเรียกใช้ฟังก์ชันกระบวนการ MediaPipe Hands เพื่อจับภาพในแต่ละเฟรมจากกล้องเว็บแคม โดยภาพแต่ละเฟรมนั้น จะแสดงผลจุด 3 มิติสำหรับแต่ละมือที่ตรวจพบ
        1.เช็คตำแหน่งของมือที่ตรวจพบได้จากกล้อง Webcam 
        2.เก็บค่าพิกัด x และ y ของมือในแต่ละตำแหน่ง 
        3.เช็คพิกัดของนิ้วแต่ละนิ้วเพื่อดูว่านิ้วมีการยกขึ้นเพื่อเพิ่มจำนวนนิ้วหรือไม่
        4.วาดเส้นบอกตำแหน่งนิ้วด้วยฟังก์ชัน draw_landmarks

    สำหรับขั้นตอนต่อไป มี 2 วิธีในการทดสอบว่านิ้วถูกยกขึ้นหรือไม่
        1.สำหรับนิ้วโป้ง จะตรวจสอบค่าของพิกัด THUMB_TIP และ THUMB_IP x และเส้นบอกตำแหน่งของมือ นิ้วโป้งจะถือว่ายกขึ้นหาก _TIP อยู่ทางด้านขวาของ _IP สำหรับมือซ้าย และอยู่ตรงกันข้ามสำหรับมือขวา
        2.สำหรับ 5 นิ้วที่เหลือ จะตรวจสอบค่าของพิกัด _TIP และ _PIP y นิ้วจะถือว่ายกขึ้นหาก _TIP อยู่สูงกว่า _PIP