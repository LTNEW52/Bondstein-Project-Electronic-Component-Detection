# Testing Part 1

Getting pictures and annote them was fun at first but got boring at the end. Although making a manual dataset instead of using kaggle or existing is very very fun!

Model training was very easy, and from the result it looked like it learned very well, mAP50 98% and mAP50-95 94%. It also detected test images very well.

Next is the app.py. Streamlit is amazing, a website without coding in html/css/js! Image detection and video detection went well, although video detection gave some trouble like chossing moviepy or not. Note to self, stick to libraries, they are there for a reason.

But there is a huge problem now. It is detecting test image well, but in other settings like different room, different distance, it is not even detecting sometimes. Battery and Display are mostly confident but others are not. Clearly seems like overfitting case. I think i will try new photos in training setting.
