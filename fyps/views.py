import os

from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.conf import settings
from fyps.models import UploadedImage
from fyps.forms import UploadImageForm
from fyps.segmentation import run_segmentation,run_classification
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from PIL import Image
def resize_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    resized_image = image.resize(target_size, Image.ANTIALIAS)
    resized_image.save(image_path)

    return resized_image


def index(request):
    """medical image analysis的主页"""
    return render(request,'fyps/index.html')

@login_required
def upload_image(request):
    if request.method == 'POST' and 'upload' in request.POST:
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.cleaned_data['image']
            uploaded_image_content = uploaded_image.read()

            # Save the resized image for display
            image_path = default_storage.save('uploaded_images/{}'.format(uploaded_image.name),
                                              ContentFile(uploaded_image_content))

            # Save the original image for segmentation
            original_image_path = default_storage.save('uploaded_images/original_{}'.format(uploaded_image.name),
                                                       ContentFile(uploaded_image_content))

            # Resize the uploaded image for display
            resize_image(os.path.join(settings.MEDIA_ROOT, image_path))

            uploaded_image_url = os.path.join(settings.MEDIA_URL, image_path)
            request.session['uploaded_image_url'] = uploaded_image_url
            request.session['original_image_url'] = os.path.join(settings.MEDIA_URL, original_image_path)
            context = {'form': form, 'uploaded_image': uploaded_image_url}
            return render(request, 'fyps/upload.html', context)
    else:
        form = UploadImageForm()
    context = {'form': form}
    return render(request, 'fyps/upload.html', context)


@login_required
def analyze_image(request):
    if request.method == 'POST':
        uploaded_image_url = request.session.get('uploaded_image_url', None)
        original_image_url = request.session.get('original_image_url', None)
        if not uploaded_image_url or not original_image_url:
            return HttpResponseRedirect(reverse('fyps:image'))

        # Remove the /media/ from the beginning of the path
        uploaded_image_url = uploaded_image_url.replace('/media/', '')
        original_image_url = original_image_url.replace('/media/', '')

        image_full_path = os.path.join(settings.MEDIA_ROOT, original_image_url)

        mask_image = run_segmentation(image_full_path)
        flag1, flag2, flag3 = run_classification(image_full_path)
        mask_image_filename = 'mask.png'
        mask_path = default_storage.save(mask_image_filename, ContentFile(mask_image.tobytes()))
        result_image_url = os.path.join(settings.MEDIA_URL, 'mask.png')

        uploaded_image_url = os.path.join(settings.MEDIA_URL, uploaded_image_url)

        classification_messages = []
        if flag1:
            classification_messages.append("ground glasses")
        if flag2:
            classification_messages.append("consolidations")
        if not classification_messages:
            classification_messages.append("healthy")

        classification = " and ".join(classification_messages)

        context = {'form': UploadImageForm(), 'uploaded_image': uploaded_image_url, 'result_image': result_image_url, 'result_image_name': mask_image_filename, 'classification': classification}
        return render(request, 'fyps/upload.html', context)




