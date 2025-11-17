import keras
import tensorflow as tf

from keras import ops

class Distiller(keras.Model):

    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, 
                     x=None, 
                     y=None, 
                     y_pred=None, 
                     sample_weight=None, 
                     allow_empty=False):

        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            ops.softmax(teacher_pred / self.temperature, axis=1),
            ops.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        return self.student(x)


class MultiTeacherDistiller(Distiller):
    
    # Instead of receiveing only a teacher, this distiller 
    # receives a list of teachers. Then, we modify the compute_loss
    # method implemented on the parent class.
    def compute_loss(self, 
                     x=None, 
                     y=None, 
                     y_pred=None, 
                     sample_weight=None, 
                     allow_empty=False):
    
        # Modified part for multi teacher knowledge distillation
        teacher_pred = []
        for teacher in self.teacher:
            teacher_pred.append(teacher(x, training=False))

        # Original student loss
        student_loss = self.student_loss_fn(y, y_pred)

        losses = [ self.distillation_loss_fn(
                    ops.softmax(pred / self.temperature, axis=1),
                    ops.softmax(y_pred / self.temperature, axis=1),
                    ) * (self.temperature**2) for pred in teacher_pred ]

        # Get the mean of losses
        distillation_loss = tf.reduce_mean(losses) 

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        return loss
