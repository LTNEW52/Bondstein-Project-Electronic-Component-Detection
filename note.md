# Testing Part 1

Getting pictures and annote them was fun at first but got boring at the end. Although making a manual dataset instead of using kaggle or existing is very very fun!

Model training was very easy, and from the result it looked like it learned very well, mAP50 98% and mAP50-95 94%. It also detected test images very well.

Next is the app.py. Streamlit is amazing, a website without coding in html/css/js! Image detection and video detection went well, although video detection gave some trouble like chossing moviepy or not. Note to self, stick to libraries, they are there for a reason.

But there is a huge problem now. It is detecting test image well, but in other settings like different room, different distance, it is not even detecting sometimes. Battery and Display are mostly confident but others are not. Clearly seems like overfitting case. I think i will try new photos in training setting.

Alright, so ran with familiar environment. Result is much better, still missclassified a lot. In the images it couldnt identify motor at all, but in video it was pretty accurate overall. So I think dataset being in only similar environment became a factor.

The reason I think is validation set is only 25. Learning on 600+ image but validation on 25 hurt a lot. I should make another dataset and re train model with better validation set. Also this time, maybe YOLOv11.

# Testing Part 2

So made a new dataset, this time with 80 validation image. Also made the model YOLOv11 from YOLOv8. Maybe because of more validation, or YOLOv11 is much stronger, maybe both, the result is much better. There is still some error, but it is much more confident on what it is detecting.

I think it is ready, but will try to fine tune more. I have to update the counter for video tomorrow, also change the environment for test case and re test. If it passes the different environment, then it is ready.

Today i fixed the count, so the count is showing on the video now. It was a big hassel though, because as it was encoded in .mp4v, it was not showing. Had to use moviepy to encode properly.

Also for testing, used a totally different environment. It performed very very poorly on images, detecting most of them as battery. Although the video performed very good! Wonder why is that? Although video was much closer, let me run another test...
