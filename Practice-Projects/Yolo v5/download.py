from simple_image_download import simple_image_download as simp

response = simp.simple_image_download

keywords=["johny depp","jack sparrow"]
for kw in keywords:
 response().download(kw , 22,extensions={'.jpg', '.png','.jpeg'})