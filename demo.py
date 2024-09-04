from mash_2dgs.Demo.mash_refiner import demo as demo_refine_with_mash
from mash_2dgs.Demo.trainer import demo as demo_train
from mash_2dgs.Demo.joint_trainer import demo as demo_joint_train

if __name__ == "__main__":
    # demo_refine_with_mash()
    demo_train()
    demo_joint_train(100)
    demo_joint_train(200)
    demo_joint_train(400)
