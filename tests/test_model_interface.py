"""Fast CPU smoke tests. No model weights, database, Redis or S3 required."""
from PIL import Image
from models.base import VTONInput, normalize_category
from models.preprocessing import Preprocessor, prepare_person, prepare_garment
from models.demo_model import DemoVTONModel


def test_preprocessor_runs_on_stub_images():
    pre=Preprocessor(); pre.load()
    person_img=Image.new('RGB',(192,256),color=(200,180,160))
    garment_img=Image.new('RGB',(192,256),color=(50,50,120))
    person=pre.process_person(person_img); garment=pre.process_garment(garment_img)
    assert person.body_mask.shape==(256,192)
    assert garment.mask.shape==(256,192)
    assert len(person.torso_bbox)==4


def test_category_mapping():
    assert normalize_category('top')=='upper_body'
    assert normalize_category('jeans')=='lower_body'
    assert normalize_category('dress')=='dresses'


def test_framing_preserves_aspect_without_stretching():
    image=Image.new('RGB',(1600,900))
    framed,box=prepare_person(image)
    assert framed.size==(768,1024)
    assert box==(462,0,1137,900)


def test_garment_canvas_is_model_sized():
    garment=prepare_garment(Image.new('RGB',(500,700)))
    assert garment.size==(768,1024)


def test_demo_backend_end_to_end():
    model=DemoVTONModel(); model.load()
    out=model.run(VTONInput(
        Image.new('RGB',(300,400),(180,160,150)),
        Image.new('RGB',(200,300),(20,80,160)),
        'top','blue t-shirt',seed=42,
    ))
    assert out.result_image.size==(300,400)
    assert out.model_name=='demo'
    assert 'demo_backend_not_real_vton' in out.quality_flags
