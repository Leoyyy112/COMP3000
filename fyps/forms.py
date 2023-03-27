# forms.py
from django import forms

# class SegmentationForm(forms.Form):
#     image = forms.ImageField()



from django import forms
from .models import UploadedImage

class UploadImageForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ('image',)

    def clean_image(self):
        image = self.cleaned_data['image']
        if image.content_type not in ['image/png']:
            raise forms.ValidationError("Only PNG images are allowed.")
        return image
